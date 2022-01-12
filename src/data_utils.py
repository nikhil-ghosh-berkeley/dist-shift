from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision import transforms
from typing import Callable, Optional, List
from torch.utils.data import TensorDataset, Subset, Dataset
import numpy as np
import torch
import torchvision

import pathlib
import json
import os
from src.imagenet_downsampled import ImageNetDS
from src.pacs_dataset import SketchDS
from src.simple_utils import load_pickle
from src.read_cifar5m import CIFAR5m
from src.imagenet_names import imagenet_classes, imagenet_a_classes, imagenet_r_classes
from PIL import Image

cifar10_label_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

class CIFAR10v2(torchvision.datasets.CIFAR10):
    
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        self.transform = transform
        self.target_transform = target_transform

        # if train: 
        data = np.load(root + "/" + 'cifar102_train.npz', allow_pickle=True)
        # else: 
            # data = np.load(root + "/" + 'cifar102_test.npy', allow_pickle=True).item()
            
        self.data = data["images"]
        self.targets = data["labels"]

    def __len__(self): 
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
class NumpyDataset(Dataset):
    def __init__(self, name: str, data_dir: str = None, transform: Optional[Callable] = None) -> None:
        self.data_dir = data_dir
        self.transform = transform
        self.name = name

        print(f"reading data from {os.path.abspath(data_dir)}")
        self.data_x = np.load(os.path.join(data_dir, name + "_data.npy"))
        self.data_y = np.load(os.path.join(data_dir, name + "_labels.npy"))

    def __getitem__(self, index):
        img, target = self.data_x[index], self.data_y[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.data_x.shape[0]

def get_index_subset(data_dir: str, idx_fname: str, full_train):
    return Subset(full_train, load_pickle(os.path.join(data_dir, idx_fname)))

def get_class_subset(label: str, class_names: List[str], full_train, full_val):
    label_idx = class_names.index(label)
    train_set = Subset(
        full_train,
        (torch.Tensor(full_train.targets) == label_idx).nonzero().flatten(),
    )
    val_set = Subset(
        full_val,
        (torch.Tensor(full_val.targets) == label_idx).nonzero().flatten(),
    )
    return train_set, val_set

def get_preprocessing(dset: str, use_aug: bool = False, train: bool = False):
    # WARNING only defined for CIFAR-type datasets

    if dset.lower().startswith("clip"):
        # to add
        raise NotImplementedError

    if dset.lower().startswith("cifar10"):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
    
    elif dset.lower().startswith("cinic10"): 
        mean = (0.47889522, 0.47227842, 0.43047404)
        std = (0.24205776, 0.23828046, 0.25874835)
        
    elif dset.lower().startswith("cifar100"):
        mean = (0.5074, 0.4867, 0.4411)
        std = (0.2011, 0.1987, 0.2025)
    elif dset.lower().startswith("cifar5m"):
        mean = (0.4555, 0.4362, 0.3415)
        std = (0.2284, 0.2167, 0.2165)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    if use_aug and train:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    if dset == "sketch":
        transform = transforms.Compose([transforms.Resize(32), transform])

    return transform


def get_dataset(data_dir: str, dset: str, transform: Optional[Callable] = None):
    splitted = dset.rsplit("_", 1)
    name = splitted[0]

    # check if dset is of the form *_test or *_train
    has_partition = (len(splitted) == 2 and (splitted[1] in ["train", "test"]))

    if not has_partition:
        if name == "CIFAR5m":
            return CIFAR5m(data_dir=None, transform=transform)
        elif name in ["CIFAR10-easy", "CIFAR10-hard", "CIFAR10-neg", "CIFAR10-neg-bal"]:
            return NumpyDataset(name, data_dir=data_dir, transform=transform)
        else:
            raise ValueError("Error invalid dataset name %s" % name)

    split = splitted[1]
    train = (split == "train")

    if name == "CIFAR10":
        return CIFAR10(data_dir, train=train, transform=transform, download=True)
    if name == "CIFAR100":
        return CIFAR100(data_dir, train=train, transform=transform, download=True)
    
    if name.lower() == "cinic10": 
        cinic_directory = data_dir + '/CINIC-10'
        dataset =  torchvision.datasets.ImageFolder(cinic_directory +'/'+ split, transform=transform)
        # rand_ind = np.random.choice(len(dataset), size=(90000,), replace=False)
        return dataset

    if name == "CIFAR10v2":
        return CIFAR10v2(data_dir, train=train, transform=transform, download=True)
    
    
    if name == "CIFAR10.1":
        images, labels = load_new_test_data("v6")
        return TensorDataset(
            torch.Tensor(images).permute(0, 3, 1, 2), torch.Tensor(labels)
        )
    if name == "CIFAR10_frog":
        full = CIFAR10(data_dir, train=train, transform=transform, download=True)
        idx = (torch.Tensor(full.targets) == 6).nonzero().flatten()
        return Subset(full, idx)
    if name == "CIFAR10.1_frog":
        images, labels = load_new_test_data("v6")
        idx = np.argwhere(labels == 6).flatten()
        return Subset(
            TensorDataset(
                torch.Tensor(images).permute(0, 3, 1, 2), torch.Tensor(labels)
            ),
            idx,
        )
    if name == "CIFAR10_subset":
        dset = CIFAR10(data_dir, train=True, transform=transform, download=False)
        idx = load_pickle(os.path.join(data_dir, "subset.pkl"))
        return Subset(dset, idx)
    if name == "imagenet32":
        return ImageNetDS(data_dir, 32, train=train, transform=transform)
    if name == "sketch":
        dset = SketchDS(data_dir, train=train, transform=transform)
        labels = np.array(dset.labels)
        idx_dog = np.argwhere(labels == 1).flatten()
        idx_horse = np.argwhere(labels == 5).flatten()
        labels[idx_dog] = 5
        labels[idx_horse] = 7
        dset.labels = labels.tolist()
        return Subset(dset, np.concatenate((idx_dog, idx_horse)))
    else:
        raise ValueError("Error invalid dataset name %s" % name)


def load_new_test_data(version_string="", load_tinyimage_indices=False):
    data_path = os.path.join(os.path.dirname(__file__), "../data/")
    filename = "cifar10.1"
    if version_string == "":
        version_string = "v7"
    if version_string in ["v4", "v6", "v7"]:
        filename += "_" + version_string
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version_string))
    label_filename = filename + "_labels.npy"
    imagedata_filename = filename + "_data.npy"
    label_filepath = os.path.abspath(os.path.join(data_path, label_filename))
    imagedata_filepath = os.path.abspath(os.path.join(data_path, imagedata_filename))
    print("Loading labels from file {}".format(label_filepath))
    assert pathlib.Path(label_filepath).is_file()
    labels = np.load(label_filepath)
    print("Loading image data from file {}".format(imagedata_filepath))
    assert pathlib.Path(imagedata_filepath).is_file()
    imagedata = np.load(imagedata_filepath)
    assert len(labels.shape) == 1
    assert len(imagedata.shape) == 4
    assert labels.shape[0] == imagedata.shape[0]
    assert imagedata.shape[1] == 32
    assert imagedata.shape[2] == 32
    assert imagedata.shape[3] == 3
    if version_string == "v6" or version_string == "v7":
        assert labels.shape[0] == 2000
    elif version_string == "v4":
        assert labels.shape[0] == 2021

    if not load_tinyimage_indices:
        return imagedata, labels
    else:
        ti_indices_data_path = os.path.join(os.path.dirname(__file__), "../other_data/")
        ti_indices_filename = "cifar10.1_" + version_string + "_ti_indices.json"
        ti_indices_filepath = os.path.abspath(
            os.path.join(ti_indices_data_path, ti_indices_filename)
        )
        print("Loading Tiny Image indices from file {}".format(ti_indices_filepath))
        assert pathlib.Path(ti_indices_filepath).is_file()
        with open(ti_indices_filepath, "r") as f:
            tinyimage_indices = json.load(f)
        assert type(tinyimage_indices) is list
        assert len(tinyimage_indices) == labels.shape[0]
        return imagedata, labels, tinyimage_indices

    
def get_dataset_imagenet(val_name):  
    from torchvision import datasets
    from torchvision.transforms import transforms
    default_data_paths = {
    'imagenet_val': '/home/jupyter/dist-shift/data/imagenet-val/',
    'imagenet_a': '/home/jupyter/dist-shift/data/imagenet-a/',
    'imagenet_r': '/home/jupyter/dist-shift/data/imagenet-r/',
    'imagenet_sketch': '/home/jupyter/dist-shift/data/sketch',
    'imagenet_v2': '/home/jupyter/dist-shift/data/imagenet-v2/'
    }
    datapath = default_data_paths[val_name]
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return datasets.ImageFolder(datapath, test_transforms)