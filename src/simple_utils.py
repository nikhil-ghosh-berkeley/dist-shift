import pickle
import torch
import filelock
import logging

from cifar10_models.clip_models import ClipViTB32
from cifar10_models.Resnet import ResNet18
from cifar10_models.Densenet import DenseNet121

from os import listdir, mkdir
from os.path import isfile, join, dirname, basename, isdir

logging.getLogger("filelock").setLevel(logging.ERROR)

all_classifiers = {
    "Resnet18": ResNet18,
    "Densenet121": DenseNet121,
    "ClipViTB32": ClipViTB32
}

def list_files(path: str):
    return [f for f in listdir(path) if isfile(join(path, f))]

def load_pickle(fname, lock_dir='locks'):
    lockfname = fname + '.lock'
    if lock_dir is not None:
        lock_path = join(dirname(lockfname), lock_dir)
        if not isdir(lock_path):
            mkdir(lock_path)
        lockfname = join(lock_path, basename(lockfname))
    lock = filelock.FileLock(lockfname)

    try:
        with lock.acquire(timeout=10):
            try:
                with open(fname, 'rb') as f:
                    result = pickle.load(f)
            except FileNotFoundError:
                print("file {} does not exist".format(fname))
    except filelock.Timeout:
        print("failed to read in time")
    except Exception as e:
        print(e)

    return result

def dump_pickle(obj, fname, lock_dir='locks'):
    lockfname = fname + '.lock'
    if lock_dir is not None:
        lock_path = join(dirname(lockfname), lock_dir)
        if not isdir(lock_path):
            mkdir(lock_path)
        lockfname = join(lock_path, basename(lockfname))

    try:
        with filelock.FileLock(lockfname, timeout=10):
            with open(fname, 'wb') as f:
                pickle.dump(obj, f)
    except filelock.Timeout:
        print("failed to write in time")
    except Exception as e:
        print(e)

def load_pretrained(arch: str, work_dir: str, num_classes: int):
    if arch == "Resnet18":
        pre_path = "Resnet18_epoch=99-step=500499.ckpt"
    elif arch == "Densenet121":
        pre_path = "Densenet121_epoch=99-step=500499.ckpt"
    else:
        raise ValueError("Invalid pretrained architecture %s" % arch)

    state_dict = torch.load(join(work_dir, "pretrained", pre_path))["state_dict"]
    state_dict = {k[6:]: v for k, v in state_dict.items()}
    model = all_classifiers[arch](num_classes=1000)
    # model = all_classifiers[arch](num_classes=state_dict['num_classes'])
    model.load_state_dict(state_dict)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.linear.in_features
    model.linear = torch.nn.Linear(num_ftrs, num_classes)
    return model
