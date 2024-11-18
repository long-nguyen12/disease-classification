from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
from PIL import Image


class MyDataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class KvasirDataLoader(Dataset):
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

        category = [
            cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root, cls))
        ]
        category.sort()
        class_indices = dict((k, v) for v, k in enumerate(category))

        self.train_images_path = []
        train_images_label = []

        self.val_images_path = []
        self.val_images_label = []

        supported = [".jpg", ".JPG", ".png", ".PNG"]

        for cls in category:
            cls_path = os.path.join(root, "train", cls)
            images = [
                os.path.join(root, "train", cls, i)
                for i in os.listdir(cls_path)
                if os.path.splitext(i)[-1] in supported
            ]

            image_class = class_indices[cls]

            for img_path in images:
                self.train_images_path.append(img_path)
                self.train_images_label.append(image_class)

        print("{} images for training.".format(len(self.train_images_path)))

        for cls in category:
            cls_path = os.path.join(root, "val", cls)
            images = [
                os.path.join(root, "val", cls, i)
                for i in os.listdir(cls_path)
                if os.path.splitext(i)[-1] in supported
            ]
            image_class = class_indices[cls]

            for img_path in images:
                self.val_images_path.append(img_path)
                self.val_images_label.append(image_class)

        print("{} images for validation.".format(len(self.val_images_path)))

    def get_data_loaders(self):

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        train_dataset = MyDataSet(
            images_path=self.train_images_path,
            images_class=self.train_images_label,
            transform=self.transform,
        )
        val_dataset = MyDataSet(
            images_path=self.val_images_path,
            images_class=self.val_images_label,
            transform=self.transform,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        return train_loader, val_loader
