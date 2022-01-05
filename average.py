import pandas as pd
import numpy as np
import os
from src.simple_utils import load_pickle, dump_pickle, list_files
from scipy.special import softmax
import time
from os.path import join


# compute average and std statistics over random trials
def average(pred_dirs, pred_base="predictions", raw_dir="raw", avg_dir="avg"):
    print("starting averaging")
    start_time = time.time()
    for pred_dir in pred_dirs:
        print(pred_dir)
        raw_path = join(pred_base, pred_dir, raw_dir)
        file_list = list_files(raw_path)
        for file in file_list:
            print(file)
            dct = load_pickle(join(raw_path, file))
            avg_dct = dict()

            index = np.concatenate(dct["index"])
            label = np.concatenate(dct["label"])

            if "logits" in dct:
                print("has_logits")
                logits = np.concatenate(dct["logits"], axis=0)
                probs = softmax(logits, axis=1)
                pred = np.argmax(logits, axis=1)
                binary = np.zeros_like(logits, dtype=int)
                binary[np.arange(len(binary)), pred] = 1
                acc = label == pred

                df = pd.DataFrame(probs)
                df["index"] = index
                avg_dct["probs"] = df.groupby(by="index").mean()

                df = pd.DataFrame(binary)
                df["index"] = index
                avg_dct["freqs"] = df.groupby(by="index").mean()
            elif "acc" in dct:
                acc = np.concatenate(dct["acc"])

            # average accuracy pointwise
            df = pd.DataFrame({"index": index, "acc": acc})
            avg_dct["points"] = df.groupby(by="index").agg(["mean", "sem"])

            # average accuracy classwise
            df = pd.DataFrame({"label": label, "acc": acc})
            avg_dct["labels"] = df.groupby(by="label").agg(["mean", "sem"])

            df = pd.DataFrame({"index": index, "labels": probs[np.arange(len(label)),label]})
            avg_dct["soft_points"] = df.groupby(by="index").agg(["mean", "sem"])
            
            avg_dct["dset_mean"] = np.mean(dct["dset"])
            avg_dct["dset_sem"] = np.std(dct["dset"]) / np.sqrt(len(dct["dset"]))

            avg_path = join(pred_base, pred_dir, avg_dir)
            if not os.path.isdir(avg_path):
                os.mkdir(avg_path)
            dump_pickle(avg_dct, join(avg_path, file))

    print("done %f" % (time.time() - start_time))
