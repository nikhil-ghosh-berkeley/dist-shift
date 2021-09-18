import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Optional, List
from torch.utils.data import Subset
import numpy as np
from src.utils import get_dataset


class CIFAR10DataModule(pl.LightningDataModule):
    name = "CIFAR10"

    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 128,
        n: Optional[int] = None,
        use_aug: bool = True,
        val_names: List[str] = [],
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n = n
        self.use_aug = use_aug
        self.val_names = val_names
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

        if use_aug:
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )
        else:
            self.train_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
            )

        self.test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
        )

    def setup(self, stage: Optional[str] = None):
        full_train = get_dataset(
            self.data_dir,
            CIFAR10DataModule.name,
            transform=self.train_transform,
            train=True,
        )

        if self.n is None:
            self.train_set = full_train
        else:
            rand_ind = np.random.choice(len(full_train), size=(self.n,), replace=False)
            self.train_set = Subset(full_train, rand_ind)

        self.val_sets = []
        self.val_sets.append(
            get_dataset(
                self.data_dir,
                CIFAR10DataModule.name,
                transform=self.test_transform,
                train=False,
            )
        )

        for val_name in self.val_names:
            name, split = val_name.rsplit("_", 1)
            self.val_sets.append(
                get_dataset(
                    self.data_dir,
                    name,
                    transform=self.test_transform,
                    train=(split == "train"),
                )
            )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return [
            DataLoader(val_set, batch_size=256, shuffle=False)
            for val_set in self.val_sets
        ]
