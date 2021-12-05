import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Optional, List, Callable
from torch.utils.data import Subset
import numpy as np
from src.data_utils import get_dataset, cifar10_label_names
from src.simple_utils import load_pickle
import torch
import os

osj = os.path.join


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: str = "CIFAR10",
        data_dir: str = "./",
        batch_size: int = 128,
        n: Optional[int] = None,
        idx_fname: Optional[str] = None,
        label: Optional[str] = None,
        use_aug: bool = True,
        val_names: List[str] = [],
        preprocess_func: Optional[Callable] = None,
        seed: int = None
    ):
        super().__init__()
        assert name in ["CIFAR10", "imagenet32"]
        if label is not None:
            assert label in cifar10_label_names

        self.name = name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n = n
        self.label = label
        self.use_aug = use_aug
        self.idx_fname = idx_fname
        self.val_names = val_names

        if name == "CIFAR10":
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2471, 0.2435, 0.2616)
        elif name == "imagenet32":
            self.mean = (0.5, 0.5, 0.5)
            self.std = (0.5, 0.5, 0.5)

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

    def setup(self, stage: Optional[str] = None):
        full_train = get_dataset(
            self.data_dir,
            self.name,
            transform=self.train_transform,
            train=True,
        )

        if self.idx_fname is not None:
            full_train = Subset(
                full_train, load_pickle(osj(self.data_dir, self.idx_fname))
            )

        val_set = get_dataset(
            self.data_dir,
            self.name,
            transform=self.test_transform,
            train=False,
        )

        if self.label is not None:
            assert self.name == "CIFAR10"
            label_idx = cifar10_label_names.index(self.label)
            full_train = Subset(
                full_train,
                (torch.Tensor(full_train.targets) == label_idx).nonzero().flatten(),
            )
            val_set = Subset(
                val_set,
                (torch.Tensor(val_set.targets) == label_idx).nonzero().flatten(),
            )

        if self.n is None:
            self.train_set = full_train
        else:
            if self.seed is not None:
                np.random.seed(self.seed)
            rand_ind = np.random.choice(len(full_train), size=(self.n,), replace=False)
            self.train_set = Subset(full_train, rand_ind)

        self.val_sets = []
        self.val_sets.append(val_set)

        for val_name in self.val_names:
            print(val_name)
            name, split = val_name.rsplit("_", 1)
            assert split in ["train", "test"]

            transform = self.test_transform
            if name == "sketch":
                transform = transforms.Compose([transforms.Resize(32), transform])

            self.val_sets.append(
                get_dataset(
                    self.data_dir,
                    name,
                    transform=transform,
                    train=(split == "train"),
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
                val_set, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True
            )
            for val_set in self.val_sets
        ]
