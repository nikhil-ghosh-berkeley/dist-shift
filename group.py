from typing import Dict
import numpy as np
import os
from src.simple_utils import load_pickle, dump_pickle, list_files
from collections import defaultdict
from os.path import join, basename

# from hash get relevent grouping attribute (e.g. architecture)
def arch_plot_group(hash: str, hash_dict: Dict) -> str:
    pg = hash_dict[hash]["arch"]
    if hash_dict[hash].get("pretrained", False):
        pg += "-pretrained"
    return pg

# plot_groups[name][pg] contains ordered list of file names (name, hash, epoch, step)
# such that plot_group(hash) = pg. (ex: name = CIFAR10_test and pg = Resnet18)
def collect_and_sort(workdir, pred_dirs, plot_group, pred_base="predictions", avg_dir="avg"):
    pred_dirs = sorted(pred_dirs)

    # collect all full raw dictionary paths
    file_list = []
    for pred_dir in pred_dirs:
        avg_path = join(workdir, pred_base, pred_dir, avg_dir)
        file_list.extend([join(avg_path, file) for file in list_files(avg_path)])

    # collect all hash_dict items from each pred_dir
    hash_dicts = [
        load_pickle(join(workdir, pred_base, pred_dir, "hash_dict.pkl")) for pred_dir in pred_dirs
    ]
    hash_dict = {k: v for dct in hash_dicts for k, v in dct.items()}

    # collect each pred_dir by plot_group
    plot_groups = dict()
    for file in file_list:
        tup = eval(basename(file))
        name, hash, _, _ = tup
        if name not in plot_groups:
            plot_groups[name] = defaultdict(list)
        pg = plot_group(hash, hash_dict)
        plot_groups[name][pg].append(file)

    # sort pred_dir in each plot_group by test-accuracy
    for name in plot_groups:
        for pg in plot_groups[name]:
            x = []
            for file in plot_groups[name][pg]:
                dct = load_pickle(file)
                x.append(dct["dset_mean"])
            ind = np.argsort(x)
            plot_groups[name][pg] = [plot_groups[name][pg][i] for i in ind]

    return plot_groups


# utilities for grouping averaged data together for plotting
# plot_group is a function taking hash and outputting plot group (str)
def group(workdir, pred_dirs, plot_group, pred_base="predictions", avg_dir="avg", proc_dir="processed"):
    print("grouping")
    pred_dirs = sorted(pred_dirs)
    plot_groups = collect_and_sort(workdir, pred_dirs, plot_group, pred_base, avg_dir)

    # group averaged data together
    grouped = dict()
    for name in plot_groups:
        print(name)
        if name not in grouped:
            grouped[name] = dict()
        for pg in plot_groups[name]:
            print(pg)
            x, x_err = [], []
            points, points_err = [], []
            soft_points, soft_points_err = [], []
            labels, labels_err = [], []
            probs, freqs = [], []

            for file in plot_groups[name][pg]:
                dct = load_pickle(file)
                x.append(dct["dset_mean"])
                x_err.append(dct["dset_sem"])

                acc_mean = ("acc", "mean")
                acc_sem = ("acc", "sem")
                
                soft_mean = ("labels", "mean")
                soft_sem = ("labels", "sem")

                df = dct["points"]
                points.append(df[acc_mean].to_numpy())
                points_err.append(df[acc_sem].to_numpy())

                df = dct["labels"]
                labels.append(df[acc_mean].to_numpy())
                labels_err.append(df[acc_sem].to_numpy())
                
                df = dct["soft_points"]
                soft_points.append(df[soft_mean].to_numpy())
                soft_points_err.append(df[soft_sem].to_numpy())

                if 'probs' in dct:
                    probs.append(np.expand_dims(dct['probs'].to_numpy(), -1))
                    freqs.append(np.expand_dims(dct['freqs'].to_numpy(), -1))

            # length len(plot_groups[name][pg])
            x = np.array(x)
            x_err = np.array(x_err)

            # points[i] is the avg points accuracies for index i
            # dimension |indexes| x len(plot_groups[name][pg])
            points = np.column_stack(points)
            points_err = np.column_stack(points_err)

            # labels[i] is the avg class accuracies for class i
            # dimension |num_classes| x len(plot_groups[name][pg])
            labels = np.column_stack(labels)
            labels_err = np.column_stack(labels_err)

            grouped[name][pg] = {
                "x": x,
                "x_err": x_err,
                "points": points,
                "points_err": points_err,
                "soft_points": soft_points,
                "soft_points_err": soft_points_err,
                "labels": labels,
                "labels_err": labels_err,
            }

            if len(probs) > 0:
                # dimension |indexes| x |num_logits| x len(plot_groups[name][pg])
                probs = np.concatenate(probs, -1)
                print("points", points.shape)
                print("labels", labels.shape)
                print("probs", probs.shape)
                grouped[name][pg]["probs"] = probs
                grouped[name][pg]["freqs"] = np.concatenate(freqs, -1)

    if not os.path.isdir(join(workdir, proc_dir)):
        os.mkdir(join(workdir, proc_dir))
    dump_pickle(grouped, join(workdir, proc_dir, "+".join(pred_dirs) + ".pkl"))
    print("done")
