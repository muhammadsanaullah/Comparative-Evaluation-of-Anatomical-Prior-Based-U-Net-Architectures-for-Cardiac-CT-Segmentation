import os

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import segmentation_models_pytorch as smp

from monai.metrics import DiceMetric

from torch.utils.data import DataLoader

from skimage.transform import resize

from dataset_2d import Cardiac2DDataset


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

BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-4

IMG_SIZE = 256


# -----------------------------------
# CREATE OUTPUT FOLDERS
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
    "ts_seg_predictions/2d_unet",
    exist_ok=True
)


# -----------------------------------
# DATASETS
# -----------------------------------

train_dataset = Cardiac2DDataset(
    split_name="train",
    image_dir="ts_data_processed/images",
    label_dir="ts_data_processed/labels_gt",
    img_size=IMG_SIZE
)

val_dataset = Cardiac2DDataset(
    split_name="val",
    image_dir="ts_data_processed/images",
    label_dir="ts_data_processed/labels_gt",
    img_size=IMG_SIZE
)

print(
    f"\nTrain slices: {len(train_dataset)}"
)

print(
    f"Validation slices: {len(val_dataset)}"
)


# -----------------------------------
# DATALOADERS
# -----------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=32,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=32,
    pin_memory=True
)


# -----------------------------------
# MODEL
# -----------------------------------

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=NUM_CLASSES
).to(DEVICE)

print("\nModel initialized.")


# -----------------------------------
# LOSS + OPTIMIZER
# -----------------------------------

loss_fn = smp.losses.DiceLoss(
    mode="multiclass"
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)


dice_metric = DiceMetric(
    include_background=False,
    reduction="mean"
)


# -----------------------------------
# TRAINING TRACKERS
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

    # -----------------------------------
    # TRAINING
    # -----------------------------------

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

        outputs = model(images)

        loss = loss_fn(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 50 == 0:

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

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            preds = torch.argmax(
                outputs,
                dim=1
            )

            dice_metric(
                preds.unsqueeze(1),
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
            "ts_seg_checkpoints/best_2d_unet.pth"
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
    "ts_seg_checkpoints/final_2d_unet.pth"
)

print("\nSaved final model")


# -----------------------------------
# SAVE TRAINING LOG
# -----------------------------------

df = pd.DataFrame({

    "epoch": list(range(1, EPOCHS + 1)),

    "train_loss": train_losses,

    "val_dice": val_dices
})

df.to_csv(
    "ts_seg_logs/2d_unet_training_log.csv",
    index=False
)

print("\nSaved training log")


# -----------------------------------
# LOSS PLOT
# -----------------------------------

plt.figure()

plt.plot(train_losses)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("2D U-Net Training Loss")

plt.savefig(
    "ts_seg_plots/2d_unet_loss.png"
)

plt.close()


# -----------------------------------
# DICE PLOT
# -----------------------------------

plt.figure()

plt.plot(val_dices)

plt.xlabel("Epoch")

plt.ylabel("Dice")

plt.title("2D U-Net Validation Dice")

plt.savefig(
    "ts_seg_plots/2d_unet_val_dice.png"
)

plt.close()

print("\nSaved plots")


# -----------------------------------
# LOAD BEST MODEL
# -----------------------------------

checkpoint = torch.load(
    "ts_seg_checkpoints/best_2d_unet.pth"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nLoaded best model")


# -----------------------------------
# TEST SUBJECTS
# -----------------------------------

with open(
    "dataset_split_ts_seg/test.txt"
) as f:

    test_subjects = [
        x.strip()
        for x in f.readlines()
    ]


# -----------------------------------
# TEST PREDICTIONS
# -----------------------------------

for subject in test_subjects:

    print(
        f"\nPredicting {subject}"
    )

    img_path = (
        f"ts_data_processed/images/{subject}.nii.gz"
    )

    nii = nib.load(img_path)

    volume = nii.get_fdata()

    prediction_volume = np.zeros(
        volume.shape,
        dtype=np.uint8
    )

    for z in range(volume.shape[2]):

        image = volume[:, :, z]

        image_resized = resize(
            image,
            (IMG_SIZE, IMG_SIZE),
            order=1,
            preserve_range=True,
            anti_aliasing=True
        )

        tensor = torch.tensor(
            image_resized,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            output = model(tensor)

            pred = torch.argmax(
                output,
                dim=1
            ).squeeze().cpu().numpy()

        pred = resize(
            pred,
            image.shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False
        )

        prediction_volume[:, :, z] = pred.astype(np.uint8)

    pred_nii = nib.Nifti1Image(
        prediction_volume,
        affine=nii.affine
    )

    save_path = (
        f"ts_seg_predictions/2d_unet/{subject}.nii.gz"
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