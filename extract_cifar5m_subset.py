from os.path import join
from src.simple_utils import load_pickle
from src.data_utils import get_dataset
import numpy as np

workdir = "/n/holystore01/LABS/barak_lab/Lab/nghosh"
data_dir = "data"
proc_dir = "processed"
name = "CIFAR5m"
pred_dir = "cifar5m"

score_types = ["MaxDiffScore", "AreaScore"]
score_type = score_types[1]

score_dir = "scores"
score_path = join(workdir, proc_dir, score_dir, pred_dir + "_" + score_type + ".pkl")
print(score_path)
scores = load_pickle(score_path)

n = 10000
idx_full = scores[name]["idx"]
num_classes = 10
dataset = get_dataset(join(workdir, data_dir), name)

# class_idxs = []
# data_y = dataset.data_y
# classes = data_y[idx_full]

# ex_per_class = n // num_classes
# for k in range(num_classes):
#     class_idx = np.where(classes == k)[0]
#     class_idxs.append(idx_full[class_idx[:ex_per_class]])

# idx = np.concatenate(class_idxs)
# print(len(idx))
# print(data_y[idx])

idx = idx_full[-n:]
dataset.extract_subset(join(workdir, data_dir), idx, "CIFAR-easy")
