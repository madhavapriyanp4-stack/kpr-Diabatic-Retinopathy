"""
APTOS 2019 dataset loader.
Expects the standard Kaggle layout:
  train.csv          -> columns: id_code, diagnosis (0-4)
  train_images/*.png
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from preprocessing import preprocess_fundus

STAGE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]


class APTOSDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, size: int = 380, transform=None, train: bool = True):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.size = size
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_fundus(img, size=self.size)

        if self.transform:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        label = int(row["diagnosis"]) if "diagnosis" in row else -1
        return img, label


def get_transforms(size: int = 380, train: bool = True):
    """Light augmentation via albumentations; falls back to plain tensors if unavailable."""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        if train:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=25, p=0.6),
                A.RandomBrightnessContrast(0.15, 0.15, p=0.4),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    except ImportError:
        return None
