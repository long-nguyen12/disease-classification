from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os


class DiseaseDataloader(Dataset):
    def __init__(
        self, root, batch_size=32, img_size=224, num_workers=4, transform=None
    ):
        self.root = root
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

        if not os.path.exists(self.root):
            raise ValueError(f"Data directory {self.root} does not exist.")

        self.dataset = datasets.ImageFolder(root=self.root, transform=self.transform)

    def get_data_loaders(self, train_val_split):
        train_size = int(train_val_split * len(self.dataset))
        val_size = len(self.dataset) - train_size

        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader, val_loader
