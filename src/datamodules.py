import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, List, Callable
from torch.utils.data import Subset
import numpy as np
from torchvision.transforms import transforms

from src.data_utils import get_dataset, get_preprocessing


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_name: str = "CIFAR10_train",
        val_names: List[str] = [],
        data_dir: str = "./",
        batch_size: int = 128,
        use_aug: bool = False,
        n: Optional[int] = None,
        preprocess_func: Optional[Callable] = None,
        seed: int = None,
        name="CIFAR10"

    ):
        super().__init__()
        self.train_name = train_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_aug = use_aug
        self.n = n
        self.val_names = val_names


        self.seed = seed
        if name == "CIFAR10":
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2471, 0.2435, 0.2616)
        elif name == "imagenet32":
            self.mean = (0.5, 0.5, 0.5)
            self.std = (0.5, 0.5, 0.5)
        elif name .lower() == 'cifar100':
            self.mean, self.std = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))

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

        if preprocess_func is not None:
            self.train_transform = preprocess_func
            self.test_transform = preprocess_func

        self.seed = seed
        self.train_transform = get_preprocessing(train_name, use_aug, train=True)
        self.test_transforms = [
            get_preprocessing(val_name, use_aug, train=False) for val_name in val_names
        ]

    def setup(self, stage: Optional[str] = None):
        full_train = get_dataset(
            self.data_dir,
            self.train_name,
            transform=self.train_transform,
        )

        if self.n is None:
            self.train_set = full_train
        else:
            rand_ind = np.random.choice(len(full_train), size=(self.n,), replace=False)
            self.train_set = Subset(full_train, rand_ind)

        self.val_sets = []
        for i, val_name in enumerate(self.val_names):
            print(val_name)
            self.val_sets.append(
                get_dataset(
                    self.data_dir,
                    val_name,
                    transform=self.test_transforms[i],
                )
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                val_set, batch_size=512, shuffle=False, num_workers=4, pin_memory=True
            )
            for val_set in self.val_sets
        ]
