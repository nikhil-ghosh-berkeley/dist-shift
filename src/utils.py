import pickle
from typing import Callable, Optional
import yaml
import hashlib
import time
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torchvision.datasets import CIFAR10
import importlib


class Categorizer:
    def __init__(self, obj_list=[]) -> None:
        self.obj_list = obj_list.copy()
        self.obj_dict = {obj: ind for ind, obj in enumerate(obj_list)}

    def add(self, obj):
        if obj not in self.obj_dict:
            self.obj_dict[obj] = len(self.obj_list)
            self.obj_list.append(obj)

    def get_obj(self, ind):
        return self.obj_list[ind]

    def get_ind(self, obj):
        return self.obj_dict[obj]

    def contains(self, obj):
        return obj in self.obj_dict

    def __repr__(self) -> str:
        return self.obj_list.__repr__()


def load_pickle(path):
    return pickle.load(open(path, "rb"))


def dump_pickle(obj, path):
    return pickle.dump(obj, open(path, "wb"))


def get_dict_hash(dictionary: dict) -> str:
    dhash = hashlib.md5()
    dump = yaml.dump(dictionary)
    encoded = dump.encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def get_class_name(module_class_string, split=None):
    module_name, class_name = module_class_string.rsplit(".", 1)
    module = importlib.import_module(module_name)
    assert hasattr(module, class_name), "class {} is not in {}".format(
        class_name, module_name
    )
    cls = getattr(module, class_name)
    name = cls.name
    if split is not None:
        name += "_" + split
    return name


def load_config(config_file: str) -> dict:
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.Loader)

    return config


def get_random_seed() -> int:
    return int(time.time() * 256) % (2 ** 32)


def filter_config(config: DictConfig) -> dict:
    def is_special_key(key: str) -> bool:
        return key[0] == "_" and key[-1] == "_"

    filt = {
        k: v
        for k, v in config.items()
        if not OmegaConf.is_interpolation(config, k) and not is_special_key(k)
    }

    return filt


def log_hyperparams(config: DictConfig, trainer: Trainer) -> None:
    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    for key in ["trainer", "model", "datamodule"]:
        hparams[key] = filter_config(config[key])

    trainer.logger.log_hyperparams(hparams)


def get_dataset(
    data_dir: str, name: str, transform: Optional[Callable] = None, train: bool = False
):
    if name == "CIFAR10":
        return CIFAR10(data_dir, train=train, transform=transform, download=True)

    return None
