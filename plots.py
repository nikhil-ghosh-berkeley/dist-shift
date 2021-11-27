import pandas as pd
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
import os
from src.simple_utils import load_pickle
from src.data_utils import cifar10_label_names

osj = os.path.join

data_dir = "data"
save_dir = "figures/CLIP"
pred_dir = "predictions/CLIP"
train = False
alpha = 0.5
markersize = 12

ind_types = ["points", "small", "weird"]
ind_type = ind_types[2]

print("loading data")
df = pd.read_pickle(osj(pred_dir, "preds.pkl"))
hash_dict = load_pickle(osj(pred_dir, "hash_dict.pkl"))
subset = load_pickle(osj(data_dir, "subset.pkl")) if train else np.arange(10000)

proc_points = load_pickle(osj(pred_dir, "proc_points.pkl"))
proc_labels = load_pickle(osj(pred_dir, "proc_labels.pkl"))
scores = load_pickle(osj(pred_dir, "scores.pkl"))
print("done")

print("loading CIFAR10")
dataset = CIFAR10(data_dir, train=train, transform=None, download=False)
print("done")

classes = cifar10_label_names
train_set = "CIFAR10_train"
default_sizes = {"CIFAR10_train": 50000, "CIFAR10_test": 10000}

names = df["name"].cat.categories
hashes = df["hash"].cat.categories
class_labels = df["label"].cat.categories
fmts = {'Resnet18': 'bo', 'Densenet121': 'ro', 'Resnet18-pretrained': 'c^', 'Densenet121-pretrained': 'm^', 'ClipViTB32': 'go'}


ind_fixed = [
    1022,
    1252,
    1977,
    2573,
    2977,
    4974,
    5564,
    6935,
    7170,
    7251,
    7901,
    8002,
    8042,
    9598,
    9773,
]

for name in proc_points:
    if ind_type == "weird":
        indexes = scores[name][:, 0][:15].astype(np.uint16)
    elif ind_type == "small":
        indexes = scores[name][:, 0][-15:].astype(np.uint16)
    elif ind_type == "points":
        indexes = ind_fixed

    for idx in indexes:
        print("index %d" % idx)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
        for plot_group in proc_points[name]:
            ax1.errorbar(
                proc_points[name][plot_group][idx]['x'],
                proc_points[name][plot_group][idx]['y'],
                yerr=proc_points[name][plot_group][idx]['y_err'],
                xerr=proc_points[name][plot_group][idx]['x_err'],
                fmt=fmts[plot_group],
                alpha=alpha,
                markersize=markersize,
                label=plot_group,
            )

        ax1.set_xlabel("Test acc on %s" % name, fontsize=16)
        ax1.set_ylabel("Probability point classified correctly", fontsize=16)
        ax1.set_title(
            "Train %s, point is index %d from %s" % (train_set, subset[idx], name),
            fontsize=20,
        )

        ax1.legend(loc='best')
        img, label = dataset.__getitem__(subset[idx])
        img = np.array(img)
        ax2.imshow(img, interpolation="nearest")
        ax2.set_title("Index %d, class: %s" % (subset[idx], classes[label]), fontsize=20)
        plt.savefig(osj(save_dir, ind_type, "plot_acc_" + str(idx) + ".png"))
        plt.close()

for name in proc_labels:
    for label in class_labels:
        plt.figure()
        print("label %s" % label)
        for plot_group in proc_labels[name]:
            plt.errorbar(
                proc_labels[name][plot_group][label]['x'],
                proc_labels[name][plot_group][label]['y'],
                yerr=proc_labels[name][plot_group][label]['y_err'],
                xerr=proc_labels[name][plot_group][label]['x_err'],
                fmt=fmts[plot_group],
                alpha=alpha,
                markersize=markersize,
                label=plot_group,
            )

        plt.xlabel("Test acc on %s" % name, fontsize=16)
        plt.ylabel("Accuracy on class", fontsize=16)
        plt.title(
            "Train %s, class %s from %s" % (train_set, classes[int(label)], name),
            fontsize=16,
        )

        plt.legend(loc='best')
        plt.savefig(osj(save_dir, "classes", "plot_acc_class" + label + ".png"))
        plt.close()
