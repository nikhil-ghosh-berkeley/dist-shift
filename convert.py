import pandas as pd
import numpy as np
import time
from src.simple_utils import dump_pickle
import os
from os.path import join, isdir


def convert(pred_dirs, pred_base="predictions", raw_dir="raw"):
    for pred_dir in pred_dirs:
        print("loading data: %s" % pred_dir)
        start_time = time.time()
        df = pd.read_pickle(join(pred_base, pred_dir, "preds.pkl"))
        print("done %f" % (time.time() - start_time))

        # computing the dataset test accuracies (note that this uses the point predictions)
        # hence is only valid for CIFAR10->CIFAR10 experiments
        print("dataset grouping")
        start_time = time.time()
        model_keys = ["name", "hash", "epoch_step"]
        gb = df.groupby(by=model_keys)
        dset = (
            df.groupby(by=(model_keys + ["seed"]), sort=False, observed=True)
            .acc.mean()
            .groupby(by=model_keys, sort=False, observed=True)
        )
        print("done %f" % (time.time() - start_time))

        print("reorganizing data")
        # split preds.pkl dataframe into files (name, hash, epoch, step)
        # each file contains a dict with lists of arrays containing index, label, acc
        # we use lists because we want to be able to append data when more trials are run
        # we store the test accs in key 'dset'
        start_time = time.time()
        for group_name, group in gb:
            print(group_name)
            name, hash, epoch_step = group_name
            epoch, step = map(int, epoch_step.split(","))
            dict_name = str((name, hash, epoch, step))
            dct = {"index": [], "label": [], "acc": []}
            dct["index"].append(group["index"].to_numpy())
            dct["label"].append(group["label"].to_numpy().astype(np.uint16))
            dct["acc"].append(group["acc"].to_numpy())
            dct["dset"] = dset.get_group(group_name).to_numpy()
            raw_path = join(pred_base, pred_dir, raw_dir)
            if not isdir(raw_path):
                os.mkdir(raw_path)
            dump_pickle(dct, join(raw_path, dict_name))
        print("done %f" % (time.time() - start_time))
