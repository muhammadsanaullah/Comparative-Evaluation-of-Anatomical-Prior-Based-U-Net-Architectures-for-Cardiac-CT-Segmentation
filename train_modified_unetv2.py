import os

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import AttentionUnet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from pathlib import Path


# =====================================================
# SETTINGS
# =====================================================

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


GT_CLASSES = 6

PSEUDO_CLASSES = 7

INPUT_CHANNELS = 7

BATCH_SIZE = 4

EPOCHS = 80

LR = 1e-4

TARGET_SIZE = (128,128,128)


# =====================================================
# OUTPUT FOLDERS
# =====================================================

os.makedirs(
    "ts_seg_checkpoints",
    exist_ok=True
)

os.makedirs(
    "ts_seg_logs",
    exist_ok=True
)

os.makedirs(
    "ts_seg_plots",
    exist_ok=True
)

os.makedirs(
    "ts_seg_predictions/modified_unetv2",
    exist_ok=True
)


# =====================================================
# PSEUDO LABEL MAP
# =====================================================

PSEUDO_LABELS = {

    "heart_atrium_left": 1,
    "heart_atrium_right": 2,
    "heart_ventricle_left": 3,
    "heart_ventricle_right": 4,
    "heart_myocardium": 5,
    "coronary_arteries": 6
}


# =====================================================
# DATASET
# =====================================================

class ModifiedCardiacDataset(Dataset):

    def __init__(
        self,
        split_file,
        image_dir,
        gt_dir,
        pseudo_dir
    ):

        with open(split_file) as f:

            self.subjects = [
                x.strip()
                for x in f.readlines()
            ]

        self.image_dir = Path(image_dir)

        self.gt_dir = Path(gt_dir)

        self.pseudo_dir = Path(pseudo_dir)

        print(
            f"\nLoaded "
            f"{len(self.subjects)} subjects"
        )

    def __len__(self):

        return len(self.subjects)

    def __getitem__(self, idx):

        subject = self.subjects[idx]

        # ---------------------------------
        # LOAD IMAGE
        # ---------------------------------

        image = nib.load(
            str(
                self.image_dir /
                f"{subject}.nii.gz"
            )
        ).get_fdata()

        # ---------------------------------
        # LOAD GT
        # ---------------------------------

        gt = nib.load(
            str(
                self.gt_dir /
                f"{subject}.nii.gz"
            )
        ).get_fdata()

        # ---------------------------------
        # CREATE PSEUDO MULTICLASS
        # ---------------------------------

        pseudo_multiclass = np.zeros(gt.shape,dtype=np.uint8)

        pseudo_channels = []

        for name, value in PSEUDO_LABELS.items():

            path = (
                self.pseudo_dir /
                subject /
                f"{name}.nii.gz"
            )

            if path.exists():

                mask = nib.load(
                str(path)
                ).get_fdata()

                binary_mask = (
                mask > 0
                ).astype(np.float32)

                pseudo_channels.append(
                binary_mask
                )

                pseudo_multiclass[
                mask > 0
                ] = value

            else:

                pseudo_channels.append(
                np.zeros(
                    gt.shape,
                    dtype=np.float32
                    )
                )




        # ---------------------------------
        # TO TENSORS
        # ---------------------------------

        #
        input_channels = [image]

        input_channels.extend(pseudo_channels)

        input_volume = np.stack(input_channels,axis=0)

        input_volume = torch.tensor(input_volume,dtype=torch.float32)

        gt = torch.tensor(
        gt,
        dtype=torch.float32
        )

        pseudo = torch.tensor(
            pseudo_multiclass,
            dtype=torch.float32
        )

        # ---------------------------------
        # ADD DIMS
        # ---------------------------------

        #print(type(input_volume))
        #print(input_volume.shape)

        input_volume = input_volume.unsqueeze(0)

        #print(input_volume.shape)

        gt = gt.unsqueeze(0).unsqueeze(0)

        pseudo = pseudo.unsqueeze(0).unsqueeze(0)

        # ---------------------------------
        # RESIZE
        # ---------------------------------

        input_volume = F.interpolate(
            input_volume,
            size=TARGET_SIZE,
            mode="trilinear",
            align_corners=False
        )

        gt = F.interpolate(
            gt,
            size=TARGET_SIZE,
            mode="nearest"
        )

        pseudo = F.interpolate(
            pseudo,
            size=TARGET_SIZE,
            mode="nearest"
        )

        # ---------------------------------
        # REMOVE BATCH DIM
        # ---------------------------------

        input_volume = input_volume.squeeze(0)

        gt = gt.squeeze(0).long()

        pseudo = pseudo.squeeze(0).long()

        return (input_volume, gt.squeeze(0), pseudo.squeeze(0))


# =====================================================
# DATASETS
# =====================================================

train_dataset = ModifiedCardiacDataset(

    split_file=
    "dataset_split_ts_seg/train.txt",

    image_dir=
    "ts_data_processed/images",

    gt_dir=
    "ts_data_processed/labels_gt",

    pseudo_dir=
    "ts_predictions"
)

val_dataset = ModifiedCardiacDataset(

    split_file=
    "dataset_split_ts_seg/val.txt",

    image_dir=
    "ts_data_processed/images",

    gt_dir=
    "ts_data_processed/labels_gt",

    pseudo_dir=
    "ts_predictions"
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)


# =====================================================
# MODEL
# =====================================================

encoder = AttentionUnet(

    spatial_dims=3,

    in_channels=INPUT_CHANNELS,

    out_channels=128,

    channels=(32,64,128,256),

    strides=(2,2,2)
).to(DEVICE)


# =====================================================
# MULTI HEAD MODEL
# =====================================================

class MultiTaskModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = encoder

        self.gt_head = nn.Conv3d(
            128,
            GT_CLASSES,
            kernel_size=1
        )

        self.pseudo_head = nn.Conv3d(
            128,
            PSEUDO_CLASSES,
            kernel_size=1
        )

    def forward(self, x):

        features = self.encoder(x)

        gt_out = self.gt_head(features)

        pseudo_out = self.pseudo_head(features)

        return gt_out, pseudo_out


model = MultiTaskModel().to(DEVICE)


# =====================================================
# LOSSES
# =====================================================

gt_loss_fn = DiceCELoss(
    to_onehot_y=True,
    softmax=True
)

pseudo_loss_fn = DiceCELoss(
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


# =====================================================
# TRACKERS
# =====================================================

train_losses = []

val_dices = []

best_dice = 0


# =====================================================
# TRAINING
# =====================================================

for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch+1}/{EPOCHS}"
    )

    model.train()

    epoch_loss = 0

    for batch_idx, (
        images,
        gt_labels,
        pseudo_labels
    ) in enumerate(train_loader):

        images = images.to(DEVICE)

        gt_labels = gt_labels.to(DEVICE)

        pseudo_labels = pseudo_labels.to(DEVICE)

        optimizer.zero_grad()

        gt_out, pseudo_out = model(images)

        gt_loss = gt_loss_fn(
            gt_out,
            gt_labels.unsqueeze(1)
        )

        pseudo_loss = pseudo_loss_fn(
            pseudo_out,
            pseudo_labels.unsqueeze(1)
        )

        total_loss = (
            1.0 * gt_loss
            +
            0.3 * pseudo_loss
        )

        total_loss.backward()

        optimizer.step()

        epoch_loss += total_loss.item()

        if batch_idx % 5 == 0:

            print(
                f"Batch {batch_idx}/"
                f"{len(train_loader)} | "
                f"Loss: {total_loss.item():.4f}"
            )

    epoch_loss /= len(train_loader)

    train_losses.append(epoch_loss)

    # -----------------------------------------
    # VALIDATION
    # -----------------------------------------

    model.eval()

    dice_metric.reset()

    with torch.no_grad():

        for images, gt_labels, _ in val_loader:

            images = images.to(DEVICE)

            gt_labels = gt_labels.to(DEVICE)

            gt_out, _ = model(images)

            preds = torch.argmax(
                gt_out,
                dim=1,
                keepdim=True
            )

            dice_metric(
                preds,
                gt_labels.unsqueeze(1)
            )

    val_dice = (
        dice_metric
        .aggregate()
        .item()
    )

    val_dices.append(val_dice)

    print(
        f"\nTrain Loss: "
        f"{epoch_loss:.4f}"
    )

    print(
        f"Validation Dice: "
        f"{val_dice:.4f}"
    )

    # -----------------------------------------
    # SAVE BEST MODEL
    # -----------------------------------------

    if val_dice > best_dice:

        best_dice = val_dice

        torch.save(
            model.state_dict(),
            "ts_seg_checkpoints/best_modified_unetv2.pth"
        )

        print(
            "\nSaved best model"
        )


# =====================================================
# SAVE FINAL MODEL
# =====================================================

torch.save(
    model.state_dict(),
    "ts_seg_checkpoints/final_modified_unetv2.pth"
)


# =====================================================
# SAVE LOG
# =====================================================

df = pd.DataFrame({

    "epoch":
    list(range(1,EPOCHS+1)),

    "train_loss":
    train_losses,

    "val_dice":
    val_dices
})

df.to_csv(
    "ts_seg_logs/modified_unetv2_log.csv",
    index=False
)


# =====================================================
# PLOTS
# =====================================================

plt.figure()

plt.plot(train_losses)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Modified U-Net Loss")

plt.savefig(
    "ts_seg_plots/modified_unetv2_loss.png"
)

plt.close()


plt.figure()

plt.plot(val_dices)

plt.xlabel("Epoch")

plt.ylabel("Dice")

plt.title("Modified U-Net Dice")

plt.savefig(
    "ts_seg_plots/modified_unetv2_dice.png"
)

plt.close()


# =====================================================
# LOAD BEST MODEL
# =====================================================

model.load_state_dict(

    torch.load(
        "ts_seg_checkpoints/best_modified_unetv2.pth"
    )
)

model.eval()


# =====================================================
# TEST PREDICTIONS
# =====================================================

with open(
    "dataset_split_ts_seg/test.txt"
) as f:

    test_subjects = [
        x.strip()
        for x in f.readlines()
    ]


for subject in test_subjects:

    print(
        f"\nPredicting {subject}"
    )

    nii = nib.load(

        f"ts_data_processed/images/"
        f"{subject}.nii.gz"
    )

    volume = nii.get_fdata()

    original_shape = volume.shape

    input_channels = [volume]

    for name in PSEUDO_LABELS.keys():

        path = (
            Path("ts_predictions")
            / subject
            / f"{name}.nii.gz"
        )

        if path.exists():

            mask = nib.load(
                str(path)
            ).get_fdata()

            mask = (
                mask > 0
            ).astype(np.float32)

        else:

            mask = np.zeros(
                volume.shape,
                dtype=np.float32
            )

        input_channels.append(mask)

    input_volume = np.stack(
        input_channels,
        axis=0
    )

    image_tensor = torch.tensor(
        input_volume,
        dtype=torch.float32
    ).unsqueeze(0)

    image_tensor = F.interpolate(
        image_tensor,
        size=TARGET_SIZE,
        mode="trilinear",
        align_corners=False
    )

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        gt_out, _ = model(image_tensor)

        pred = torch.argmax(
            gt_out,
            dim=1
        ).float()

    pred = F.interpolate(
        pred.unsqueeze(1),
        size=original_shape,
        mode="nearest"
    )

    pred = pred.squeeze().cpu().numpy()

    pred_nii = nib.Nifti1Image(
        pred.astype(np.uint8),
        affine=nii.affine
    )

    save_path = (
        f"ts_seg_predictions/"
        f"modified_unetv2/"
        f"{subject}.nii.gz"
    )

    nib.save(
        pred_nii,
        save_path
    )

    print(
        f"Saved: {save_path}"
    )


print("\nFinished.")