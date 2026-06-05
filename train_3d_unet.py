import os

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

from torch.utils.data import DataLoader

from dataset3dunet import Cardiac3DDataset


# -----------------------------------
# SETTINGS
# -----------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing device: {DEVICE}")

if DEVICE == "cuda":

    print(
        torch.cuda.get_device_name(0)
    )


NUM_CLASSES = 6

BATCH_SIZE = 4
EPOCHS = 60
LR = 1e-4


# -----------------------------------
# OUTPUT FOLDERS
# -----------------------------------

os.makedirs(
    "ts_seg_checkpoints",
    exist_ok=True
)

os.makedirs(
    "ts_seg_plots",
    exist_ok=True
)

os.makedirs(
    "ts_seg_logs",
    exist_ok=True
)

os.makedirs(
    "ts_seg_predictions/3d_unet",
    exist_ok=True
)


# -----------------------------------
# DATASETS
# -----------------------------------

train_dataset = Cardiac3DDataset(
    split_file="dataset_split_ts_seg/train.txt",
    image_dir="ts_data_processed/images",
    label_dir="ts_data_processed/labels_gt",
    target_size=(128, 128, 128)
)

val_dataset = Cardiac3DDataset(
    split_file="dataset_split_ts_seg/val.txt",
    image_dir="ts_data_processed/images",
    label_dir="ts_data_processed/labels_gt",
    target_size=(128, 128, 128)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)


# -----------------------------------
# MODEL
# -----------------------------------

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=NUM_CLASSES,
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2),
    num_res_units=2
).to(DEVICE)


# -----------------------------------
# LOSS
# -----------------------------------

loss_fn = DiceLoss(
    to_onehot_y=True,
    softmax=True
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

dice_metric = DiceMetric(
    include_background=False,
    reduction="mean"
)

scaler = torch.cuda.amp.GradScaler()


# -----------------------------------
# TRACKERS
# -----------------------------------

train_losses = []

val_dices = []

best_dice = 0


# -----------------------------------
# TRAINING LOOP
# -----------------------------------

for epoch in range(EPOCHS):

    print(
        f"\nStarting Epoch "
        f"{epoch+1}/{EPOCHS}"
    )

    model.train()

    epoch_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            outputs = model(images)

            loss = loss_fn(
                outputs,
                labels.unsqueeze(1)
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        epoch_loss += loss.item()

        if batch_idx % 10 == 0:

            print(
                f"Epoch {epoch+1} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss /= len(train_loader)

    train_losses.append(epoch_loss)

    # -----------------------------------
    # VALIDATION
    # -----------------------------------

    model.eval()

    dice_metric.reset()

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            preds = torch.argmax(
                outputs,
                dim=1,
                keepdim=True
            )

            dice_metric(
                preds,
                labels.unsqueeze(1)
            )

    val_dice = dice_metric.aggregate().item()

    val_dices.append(val_dice)

    print(
        f"\nEpoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {epoch_loss:.4f} | "
        f"Val Dice: {val_dice:.4f}"
    )

    # -----------------------------------
    # SAVE BEST MODEL
    # -----------------------------------

    if val_dice > best_dice:

        best_dice = val_dice

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_dice
            },
            "ts_seg_checkpoints/best_3d_unet.pth"
        )

        print("\nSaved best model")


# -----------------------------------
# SAVE FINAL MODEL
# -----------------------------------

torch.save(
    {
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_dice": best_dice
    },
    "ts_seg_checkpoints/final_3d_unet.pth"
)


# -----------------------------------
# SAVE LOG
# -----------------------------------

df = pd.DataFrame({

    "epoch": list(range(1, EPOCHS + 1)),

    "train_loss": train_losses,

    "val_dice": val_dices
})

df.to_csv(
    "ts_seg_logs/3d_unet_training_log.csv",
    index=False
)


# -----------------------------------
# PLOTS
# -----------------------------------

plt.figure()

plt.plot(train_losses)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("3D U-Net Training Loss")

plt.savefig(
    "ts_seg_plots/3d_unet_loss.png"
)

plt.close()


plt.figure()

plt.plot(val_dices)

plt.xlabel("Epoch")

plt.ylabel("Dice")

plt.title("3D U-Net Validation Dice")

plt.savefig(
    "ts_seg_plots/3d_unet_val_dice.png"
)

plt.close()


# -----------------------------------
# LOAD BEST MODEL
# -----------------------------------

checkpoint = torch.load(
    "ts_seg_checkpoints/best_3d_unet.pth"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# -----------------------------------
# TEST PREDICTIONS
# -----------------------------------

with open(
    "dataset_split_ts_seg/test.txt"
) as f:

    test_subjects = [
        x.strip()
        for x in f.readlines()
    ]


TARGET_SIZE = (128, 128, 128)


for subject in test_subjects:

    print(
        f"\nPredicting {subject}"
    )

    img_path = (
        f"ts_data_processed/images/{subject}.nii.gz"
    )

    nii = nib.load(img_path)

    volume = nii.get_fdata()

    original_shape = volume.shape

    # -----------------------------------
    # TO TENSOR
    # -----------------------------------

    image_tensor = torch.tensor(
        volume,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    # -----------------------------------
    # RESIZE INPUT
    # -----------------------------------

    image_tensor = torch.nn.functional.interpolate(
        image_tensor,
        size=TARGET_SIZE,
        mode="trilinear",
        align_corners=False
    )

    image_tensor = image_tensor.to(DEVICE)

    # -----------------------------------
    # PREDICT
    # -----------------------------------

    with torch.no_grad():

        output = model(image_tensor)

        pred = torch.argmax(
            output,
            dim=1
        ).float()

    # -----------------------------------
    # RESIZE BACK
    # -----------------------------------

    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=original_shape,
        mode="nearest"
    )

    pred = pred.squeeze().cpu().numpy()

    # -----------------------------------
    # SAVE
    # -----------------------------------

    pred_nii = nib.Nifti1Image(
        pred.astype(np.uint8),
        affine=nii.affine
    )

    save_path = (
        f"ts_seg_predictions/3d_unet/{subject}.nii.gz"
    )

    nib.save(
        pred_nii,
        save_path
    )

    print(
        f"Saved prediction: "
        f"{save_path}"
    )


print("\nFinished.")