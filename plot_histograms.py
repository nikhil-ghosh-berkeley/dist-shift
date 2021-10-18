import pandas as pd
import numpy as np
import os
from src.utils import load_pickle
import matplotlib.pyplot as plt

osj = os.path.join
pred_dir1 = "predictions_early_stopping"
pred_dir2 = "predictions_early_stopping_train"

df1 = pd.read_pickle(osj(pred_dir1, "preds.pkl"))
df2 = pd.read_pickle(osj(pred_dir2, "preds.pkl"))

names1 = df1["name"].cat.categories.to_list()
names2 = df2["name"].cat.categories.to_list()

scores1 = load_pickle(osj(pred_dir1, "scores.pkl"))
scores2 = load_pickle(osj(pred_dir2, "scores.pkl"))

alpha = 0.5


for name1 in names1:
    for name2 in names2:
        plt.figure()
        s1 = scores1[name1][:, 1]
        s2 = scores2[name2][:, 1]
        plt.hist(s1, bins='sqrt', label=name1, alpha=alpha, density=False)
        plt.hist(s2, bins='sqrt', label=name2, alpha=alpha, density=False)
        plt.xlabel("Non-monotonicity score")
        plt.ylabel("Number of points")
        plt.legend()
        plt.savefig(osj("figures", "scores_compare_hist.png"))
        plt.close()

for name1 in names1:
    for name2 in names2:
        labels = [name1, name2]
        s1 = scores1[name1][:, 1]
        s2 = scores2[name2][:, 1]
        plt.boxplot([s1, s2], labels=labels, meanline=True, showfliers=True)
        plt.ylabel('Non-monotonicity score')
        plt.savefig(osj("figures", "scores_compare_box_outliers.png"))
        plt.close()

