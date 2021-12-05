import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import itertools

osj = os.path.join

pred_dirs = ["early_stopping", "pretrained", "CLIP"]
df_list = [pd.read_pickle(osj("predictions", pred_dir, "preds.pkl")) for pred_dir in pred_dirs]
names = [df["name"].cat.categories.to_list() for df in df_list]
scores = [pickle.load(open(osj("predictions", pred_dir, "scores.pkl"), "rb")) for pred_dir in pred_dirs]
alpha = 0.5

for name_tup in itertools.product(*names):
    plt.figure()
    for i in range(len(name_tup)):
        s = scores[i][name_tup[i]][:, 1]
        plt.hist(s, bins=50, label=pred_dirs[i], alpha=alpha, density=False, range=[0, 0.5])
    plt.xlabel("Non-monotonicity score")
    plt.ylabel("Number of points")
    plt.title("Score on CIFAR10_test")
    plt.legend()
    plt.savefig(osj("figures", "scores_compare_hist_pretrained.png"))
    plt.close()

for name_tup in itertools.product(*names):
    plt.figure()
    s = [scores[i][name_tup[i]][:, 1] for i in range(len(name_tup))]
    plt.boxplot(s, labels=pred_dirs, meanline=True, showfliers=False)
    plt.ylabel("Non-monotonicity score")
    plt.title("Score on CIFAR10_test")
    plt.savefig(osj("figures", "scores_compare_box_pretrained.png"))
    plt.close()
