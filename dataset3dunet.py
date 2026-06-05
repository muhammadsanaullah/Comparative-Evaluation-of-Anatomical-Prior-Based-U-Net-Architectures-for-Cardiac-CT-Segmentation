from pathlib import Path

import nibabel as nib

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset


class Cardiac3DDataset(Dataset):

    def __init__(
        self,
        split_file,
        image_dir,
        label_dir,
        target_size=(128, 128, 128)
    ):

        with open(split_file) as f:

            self.subjects = [
                x.strip()
                for x in f.readlines()
            ]

        self.image_dir = Path(image_dir)

        self.label_dir = Path(label_dir)

        self.target_size = target_size

        print(
            f"\nLoaded "
            f"{len(self.subjects)} subjects"
        )

    def __len__(self):

        return len(self.subjects)

    def __getitem__(self, idx):

        subject = self.subjects[idx]

        # -----------------------------------
        # PATHS
        # -----------------------------------

        img_path = (
            self.image_dir /
            f"{subject}.nii.gz"
        )

        lbl_path = (
            self.label_dir /
            f"{subject}.nii.gz"
        )

        # -----------------------------------
        # LOAD NIFTI
        # -----------------------------------

        image = nib.load(
            str(img_path)
        ).get_fdata()

        label = nib.load(
            str(lbl_path)
        ).get_fdata()

        # -----------------------------------
        # TO TENSOR
        # -----------------------------------

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        # -----------------------------------
        # ADD CHANNEL + BATCH DIMS
        # -----------------------------------

        image = image.unsqueeze(0).unsqueeze(0)

        label = label.unsqueeze(0).unsqueeze(0)

        # -----------------------------------
        # FORCE SIZE DIVISIBLE BY 16
        # VERY IMPORTANT FOR UNET
        # -----------------------------------

        target_d = (
            (self.target_size[0] // 16)
            * 16
        )

        target_h = (
            (self.target_size[1] // 16)
            * 16
        )

        target_w = (
            (self.target_size[2] // 16)
            * 16
        )

        target_size = (
            target_d,
            target_h,
            target_w
        )

        # -----------------------------------
        # RESIZE IMAGE
        # -----------------------------------

        image = F.interpolate(
            image,
            size=target_size,
            mode="trilinear",
            align_corners=False
        )

        # -----------------------------------
        # RESIZE LABEL
        # -----------------------------------

        label = F.interpolate(
            label,
            size=target_size,
            mode="nearest"
        )

        # -----------------------------------
        # REMOVE BATCH DIM
        # -----------------------------------

        image = image.squeeze(0)

        label = label.squeeze(0)

        # -----------------------------------
        # LABEL TYPE
        # -----------------------------------

        label = label.long().squeeze(0)

        return image, label