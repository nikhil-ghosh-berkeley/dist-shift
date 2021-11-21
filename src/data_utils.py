from torchvision.datasets import CIFAR10
from typing import Callable, Optional
from torch.utils.data import TensorDataset, Subset
import numpy as np
import torch
from src.simple_utils import load_pickle
import pathlib
import json
import sys
import os

cifar10_label_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def get_dataset(
    data_dir: str, name: str, transform: Optional[Callable] = None, train: bool = False
):
    if name == "CIFAR10":
        return CIFAR10(data_dir, train=train, transform=transform, download=True)
    if name == "CIFAR10.1":
        images, labels = load_new_test_data("v6")
        return TensorDataset(torch.Tensor(images).permute(0, 3, 1, 2), torch.Tensor(labels))
    if name == "CIFAR10_frog":
        full = CIFAR10(data_dir, train=train, transform=transform, download=True)
        idx = (torch.Tensor(full.targets) == 6).nonzero().flatten()
        return Subset(full, idx)
    if name == "CIFAR10.1_frog":
        images, labels = load_new_test_data("v6")
        idx = np.argwhere(labels == 6).flatten()
        return Subset(TensorDataset(torch.Tensor(images).permute(0, 3, 1, 2), torch.Tensor(labels)), idx)
    if name == "CIFAR10_subset":
        idx = load_pickle(os.path.join(data_dir, "subset.pkl"))
        return Subset(CIFAR10(data_dir, train=True, transform=transform, download=True), idx)
    else:
        print("Error invalid dataset name")
        sys.exit(1)


def load_new_test_data(version_string='', load_tinyimage_indices=False):
    data_path = os.path.join(os.path.dirname(__file__), '../data/')
    filename = 'cifar10.1'
    if version_string == '':
        version_string = 'v7'
    if version_string in ['v4', 'v6', 'v7']:
        filename += '_' + version_string
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version_string))
    label_filename = filename + '_labels.npy'
    imagedata_filename = filename + '_data.npy'
    label_filepath = os.path.abspath(os.path.join(data_path, label_filename))
    imagedata_filepath = os.path.abspath(os.path.join(data_path, imagedata_filename))
    print('Loading labels from file {}'.format(label_filepath))
    assert pathlib.Path(label_filepath).is_file()
    labels = np.load(label_filepath)
    print('Loading image data from file {}'.format(imagedata_filepath))
    assert pathlib.Path(imagedata_filepath).is_file()
    imagedata = np.load(imagedata_filepath)
    assert len(labels.shape) == 1
    assert len(imagedata.shape) == 4
    assert labels.shape[0] == imagedata.shape[0]
    assert imagedata.shape[1] == 32
    assert imagedata.shape[2] == 32
    assert imagedata.shape[3] == 3
    if version_string == 'v6' or version_string == 'v7':
        assert labels.shape[0] == 2000
    elif version_string == 'v4':
        assert labels.shape[0] == 2021

    if not load_tinyimage_indices:
        return imagedata, labels
    else:
        ti_indices_data_path = os.path.join(os.path.dirname(__file__), '../other_data/')
        ti_indices_filename = 'cifar10.1_' + version_string + '_ti_indices.json'
        ti_indices_filepath = os.path.abspath(os.path.join(ti_indices_data_path, ti_indices_filename))
        print('Loading Tiny Image indices from file {}'.format(ti_indices_filepath))
        assert pathlib.Path(ti_indices_filepath).is_file()
        with open(ti_indices_filepath, 'r') as f:
            tinyimage_indices = json.load(f)
        assert type(tinyimage_indices) is list
        assert len(tinyimage_indices) == labels.shape[0]
        return imagedata, labels, tinyimage_indices
