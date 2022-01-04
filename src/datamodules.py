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
    ):
        super().__init__()
        self.train_name = train_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_aug = use_aug
        self.n = n
        self.val_names = val_names

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
