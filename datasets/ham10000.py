import os

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        X = Image.open(self.df["path"][index]).resize((64, 64))
        y = torch.tensor(int(self.df["target"][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


class HamDataloader(Dataset):
    def __init__(
        self,
        train_csv,
        test_csv,
        batch_size=32,
        img_size=224,
        transform=None,
        num_workers=4,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.transform = (
            transform
            if transform is not None
            else transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        )

        self.train_dataset = SkinDataset(train_csv, transform=self.transform)
        self.test_dataset = SkinDataset(test_csv, transform=self.transform)

    def get_data_loaders(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return train_loader, val_loader