import pandas as pd
import os
import time
from collections import OrderedDict
from src.utils import load_pickle
import sys

osj = os.path.join
preds_save_loc = "predictions_early_stopping_train/"

print("start reading")
read_start = time.time()
cats = load_pickle(osj(preds_save_loc, "cats.pkl"))
print(cats)
print("reading finished: elapsed %f" % (time.time() - read_start))

print("start reading")
read_start = time.time()
df_add = pd.read_csv(
    osj(preds_save_loc, "preds.csv"),
    dtype={
        "index": "uint16",
        "epoch_step": "category",
        "name": "category",
        "hash": "category",
        "seed": "category",
        "label": "category"
    },
)

for col in cats:
    df_add[col] = df_add[col].cat.rename_categories(lambda x: list(cats[col].keys())[int(x)])

print("reading finished: elapsed %f" % (time.time() - read_start))
print("committing %d entries" % len(df_add))

if os.path.isfile(osj(preds_save_loc, "preds.pkl")):
    df_old = pd.read_pickle(osj(preds_save_loc, "preds.pkl"))
    pd.to_pickle(df_old, osj(preds_save_loc, "preds_old.pkl"))
    df = pd.concat((df_old, df_add))
    pd.to_pickle(df, osj(preds_save_loc, "preds.pkl"))
else:
    pd.to_pickle(df_add, osj(preds_save_loc, "preds.pkl"))

# os.remove(osj(preds_save_loc, "cats.pkl"))
# os.remove(osj(preds_save_loc, "preds.csv"))
