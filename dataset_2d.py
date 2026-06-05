from pathlib import Path

import nibabel as nib
import pandas as pd
import torch

from skimage.transform import resize

from torch.utils.data import Dataset


class Cardiac2DDataset(Dataset):

    def __init__(
        self,
        split_name,
        image_dir,
        label_dir,
        img_size=256
    ):

        self.df = pd.read_csv(
            "slice_index.csv"
        )

        self.df = self.df[
            self.df["split"] == split_name
        ]

        self.image_dir = Path(image_dir)

        self.label_dir = Path(label_dir)

        self.img_size = img_size

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        subject = row["subject"]

        z = int(row["slice"])

        img_path = (
            self.image_dir /
            f"{subject}.nii.gz"
        )

        lbl_path = (
            self.label_dir /
            f"{subject}.nii.gz"
        )

        image = nib.load(
            str(img_path)
        ).dataobj[:, :, z]

        label = nib.load(
            str(lbl_path)
        ).dataobj[:, :, z]

        image = resize(
            image,
            (self.img_size, self.img_size),
            order=1,
            preserve_range=True,
            anti_aliasing=True
        )

        label = resize(
            label,
            (self.img_size, self.img_size),
            order=0,
            preserve_range=True,
            anti_aliasing=False
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label