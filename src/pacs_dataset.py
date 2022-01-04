import os
import h5py
import numpy as np
import torch.utils.data as data
from PIL import Image


class SketchDS(data.Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        # now load the numpy arrays
        if self.train:
            file = os.path.join(root, "PACS", "sketch", "sketch_train.hdf5")
        else:
            file = os.path.join(root, "PACS", "sketch", "sketch_val.hdf5")

        with h5py.File(file, "r") as f:
            self.data = list(f['images'])
            self.labels = list(f['labels'])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.astype('uint8'))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

