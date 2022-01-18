from skfda import FDataGrid
from skfda.preprocessing.smoothing.kernel_smoothers import KNeighborsSmoother
import numpy as np
import os
from os.path import join, isdir
from src.simple_utils import load_pickle, dump_pickle


def smoothen(x, Y):
    smoother = KNeighborsSmoother()
    fd = FDataGrid(
        grid_points=x,
        data_matrix=Y,
    )

    fd_smoothed = smoother.fit_transform(fd)
    x_smooth = fd_smoothed.grid_points[0]
    Y_smooth = np.squeeze(fd_smoothed.data_matrix, -1)

    return x_smooth, Y_smooth

class AreaScore:
    def compute_scores(self, x, Y):
        return np.trapz(x - Y, x, axis=1)

class MaxDiffScore:
    # x is a n dimensional vector
    # Y is a k x n dimensional matrix
    # t is the min gap to consider drop
    def compute_scores(self, x, Y, t=2):
        n = len(x)
        k = Y.shape[0]
        scores = np.zeros(k)

        for i in range(n):
            for j in range(i + t, n):
                diff = Y[:, i] - Y[:, j]
                scores = np.maximum(scores, diff)

        return scores

# score grouped statistics (also adding in smoothed data)
def score_group(workdir, pred_dirs, scorer, proc_dir="processed"):
    scorer_name = scorer.__class__.__name__
    print("scoring", scorer_name)
    pred_dirs = sorted(pred_dirs)
    grouped = load_pickle(join(workdir, proc_dir, "+".join(pred_dirs) + ".pkl"))
    all_scores = dict()

    for name in grouped:
        print(name)
        scores_list = []
        for pg in grouped[name]:
            print(pg)
            dct = grouped[name][pg]
            x_smooth, Y_smooth = smoothen(dct["x"], dct["points"])
            scores = scorer.compute_scores(x_smooth, Y_smooth)
            scores_list.append(scores)
            dct["x_smooth"] = x_smooth
            dct["points_smooth"] = Y_smooth

        scores_stack = np.vstack(scores_list)
        scores = scores_stack.min(axis=0)
        idx = np.argsort(-scores)
        scores = scores[idx]

        all_scores[name] = {"vals": scores, "idx": idx}

    group_dir = "+".join(pred_dirs)
    dump_pickle(grouped, join(workdir, proc_dir, group_dir + ".pkl"))
    if not isdir(join(workdir, proc_dir, "scores")):
        os.mkdir(join(workdir, proc_dir, "scores"))
    dump_pickle(all_scores, join(workdir, proc_dir, "scores", group_dir + "_" + scorer_name + ".pkl"))
    print("done")
