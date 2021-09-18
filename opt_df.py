import pandas as pd
import os
import time
from src.utils import load_pickle

osj = os.path.join
preds_save_loc = "predictions_early_stopping/"

cats = load_pickle(osj(preds_save_loc, "cats.pkl"))

print("start reading")
read_start = time.time()
df = pd.read_csv(
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
    df[col] = df[col].cat.rename_categories(lambda x: cats[col].obj_list[int(x)])

print("reading finished: elapsed %f" % (time.time() - read_start))
pd.to_pickle(df, osj(preds_save_loc, "preds.pkl"))
