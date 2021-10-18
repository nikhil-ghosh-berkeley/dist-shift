import pandas as pd
import numpy as np
import os
from src.simple_utils import load_pickle, dump_pickle
from itertools import product
import time
from skfda import FDataGrid
from skfda.preprocessing.smoothing.kernel_smoothers import KNeighborsSmoother
import matplotlib.pyplot as plt


def process_grouping(names, plot_groups, inds, grouped, group_by_model, smooth=False):
    proc = dict()
    for name in names:
        proc[name] = dict()
        for plot_group in plot_groups:
            proc[name][plot_group] = dict()
            for idx in inds:
                proc[name][plot_group][idx] = {
                    "x": [],
                    "y": [],
                    "x_err": [],
                    "y_err": [],
                }

    print("starting")
    start_time = time.time()
    for (name, hash, idx), group in grouped:
        plot_group = hash_dict[hash]["arch"]
        acc_set = group_by_model.loc[name, hash].to_numpy()
        acc_point = group.to_numpy()
        d = proc[name][plot_group][idx]
        d["x"].append(acc_set[:, 0])
        d["x_err"].append(acc_set[:, 1])
        d["y"].append(acc_point[:, 0])
        d["y_err"].append(acc_point[:, 1])
    print("done %f" % (time.time() - start_time))

    print("concatenating")
    start_time = time.time()
    smoother = KNeighborsSmoother()
    for name, plot_group, idx in product(names, plot_groups, inds):
        d = proc[name][plot_group][idx]
        for k, v in d.items():
            d[k] = np.concatenate(v)
        if smooth:
            sort_ind = np.argsort(d["x"])
            d["x"] = d["x"][sort_ind]
            d["y"] = d["y"][sort_ind]
            fd = FDataGrid(
                grid_points=d["x"],
                data_matrix=d["y"][np.newaxis, :],
            )
            fd_smoothed = smoother.fit_transform(fd)
            d["x_smooth"] = fd_smoothed.grid_points[0].flatten()
            d["y_smooth"] = fd_smoothed.data_matrix.flatten()

    print("done %f" % (time.time() - start_time))
    return proc

def compute_scores(names, plot_groups, inds, proc, t=2):
    scores = dict()
    for name in names:
        scores[name] = []
        for idx in inds:
            if idx % 1000 == 0:
                print('index %d' % idx)
            min_group = 1
            for plot_group in plot_groups:
                f = proc[name][plot_group][idx]["y_smooth"]
                n = len(f)
                max_diff = 0
                for x in range(n):
                    for y in range(x + t, n):
                        diff = f[x] - f[y]
                        max_diff = max(max_diff, diff)
                min_group = min(min_group, max_diff)
            scores[name].append([idx, min_group])
        scores[name] = np.array(scores[name])
        scores_sort = np.argsort(-scores[name][:, 1])
        scores[name] = scores[name][scores_sort, :]
    return scores

osj = os.path.join
pred_dir = "predictions_early_stopping_train"
save_dir = "figures_early_stopping_train"

print("loading data")
df = pd.read_pickle(osj(pred_dir, "preds.pkl"))
hash_dict = load_pickle(osj(pred_dir, "hash_dict.pkl"))
group_by_model = pd.read_pickle(osj(pred_dir, "group_by_model.pkl"))

point_agg = pd.read_pickle(osj(pred_dir, "point_agg.pkl"))
group_by_point = point_agg.groupby(by=["name", "hash", "index"])

label_agg = pd.read_pickle(osj(pred_dir, "label_agg.pkl"))
group_by_label = label_agg.groupby(by=["name", "hash", "label"])
print("done")

names = df["name"].cat.categories.to_list()
hashes = df["hash"].cat.categories.to_list()
class_labels = df["label"].cat.categories.to_list()
indexes = df["index"].unique().tolist()
plot_groups = set([hash_dict[hash]["arch"] for hash in hashes])

proc_points = process_grouping(
    names, plot_groups, indexes, group_by_point, group_by_model, smooth=True
)
dump_pickle(proc_points, osj(pred_dir, "proc_points.pkl"))

# proc_points = load_pickle(osj(pred_dir, "proc_points.pkl"))

scores = compute_scores(names, plot_groups, indexes, proc_points, t=2)
dump_pickle(scores, osj(pred_dir, "scores.pkl"))

for name in names:
    inds = scores[name][:, 0]
    s = scores[name][:, 1]
    print(s[:10])
    print(inds[:10])
    plt.hist(s, bins=100)

plt.savefig(osj(save_dir, "scores.png"))

proc_points = process_grouping(
    names, plot_groups, class_labels, group_by_label, group_by_model
)
dump_pickle(proc_points, osj(pred_dir, "proc_labels.pkl"))
