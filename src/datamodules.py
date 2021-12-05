import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Optional, List, Callable
from torch.utils.data import Subset
import numpy as np
from src.data_utils import get_dataset, cifar10_label_names
import torch


class CIFAR10DataModule(pl.LightningDataModule):
    name = "CIFAR10"
    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 128,
        n: Optional[int] = None,
        label: Optional[str] = None,
        use_aug: bool = True,
        val_names: List[str] = [],
        preprocess_func: Optional[Callable] = None,
        seed: int = None,
        datasetname="CIFAR10"
    ):
        super().__init__()
        if label is not None:
            assert label in cifar10_label_names

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n = n
        self.label = label
        self.use_aug = use_aug
        self.val_names = val_names
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
        self.datasetname = datasetname
        
        if datasetname.lower() == 'cifar100':
            self.mean, self.std = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))

        self.seed = seed

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

    def setup(self, stage: Optional[str] = None):
        full_train = get_dataset(
            self.data_dir,
            self.datasetname,
            transform=self.train_transform,
            train=True,
        )

        val_set = get_dataset(
            self.data_dir,
            self.datasetname,
            transform=self.test_transform,
            train=False,
        )

        if self.label is not None:
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
            name, split = val_name.rsplit("_", 1)
            assert split in ["train", "test"]
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
