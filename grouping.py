import pandas as pd
import os
import time
import numpy as np
import sys

osj = os.path.join
preds_save_loc = "predictions/CLIP/"
df = pd.read_pickle(osj(preds_save_loc, "preds.pkl"))

print("start grouping by model")
group_start = time.time()

model_keys = ["name", "hash", "epoch_step"]
# filenames = os.listdir(osj(preds_save_loc, "ds_avg"))
# arr = [eval(s[:s.index(".")]) for s in filenames]
# tuples = [('CIFAR10_test', hash, str(epoch) + ',' + str(step)) for (hash, epoch, step) in arr]
# dset_stats = np.zeros((len(filenames), 2))

# for i, file in enumerate(filenames):
#     with open(osj(preds_save_loc, "ds_avg", file)) as f:
#         accs = np.array(list(map(float, f.read().splitlines())))
#     dset_stats[i, 0] = np.mean(accs)
#     dset_stats[i, 1] = np.std(accs) / np.sqrt(len(accs))
# print(dset_stats)
# idx = pd.MultiIndex.from_tuples(tuples=tuples, names=('name', 'hash', 'epoch_step'))
# group_by_model = pd.DataFrame(dset_stats, index=idx, columns=['mean', 'sem'])

# for each dataset; fix hash, epoch_step; get mean and se over random seeds
group_by_model = (
    df.groupby(by=(model_keys + ["seed"]), sort=False, observed=True)
    .acc.mean()
    .groupby(by=model_keys, sort=False, observed=True)
    .agg(["mean", "sem"])
)

pd.to_pickle(group_by_model, osj(preds_save_loc, "group_by_model.pkl"))
print("grouping model finished: elapsed %f" % (time.time() - group_start))


print("start grouping by index")
group_start = time.time()
# for each dataset, index; fix hash, epoch_step; get mean and se over random seed
point_agg = df.groupby(by=(model_keys + ["index"]), sort=False, observed=True).acc.agg(["mean", "sem"])
pd.to_pickle(point_agg, osj(preds_save_loc, "point_agg.pkl"))
print("grouping point finished: elapsed %f" % (time.time() - group_start))

print("start grouping by label")
group_start = time.time()
label_agg = (
    df.groupby(by=(model_keys + ["label", "seed"]), sort=False, observed=True)
    .acc.mean()
    .groupby(by=(model_keys + ["label"]), sort=False, observed=True)
    .agg(["mean", "sem"])
)

# for each dataset, label; fix hash, epoch_step; get mean and se over random seed
pd.to_pickle(label_agg, osj(preds_save_loc, "label_agg.pkl"))
print("grouping label finished: elapsed %f" % (time.time() - group_start))
