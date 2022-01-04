import pandas as pd
import time
import pickle
from os.path import join

def csv_to_pickle(pred_dirs, pred_base="predictions"):
    for pred_dir in pred_dirs:
        preds_save_path = join(pred_base, pred_dir)
        print("start reading preds.csv")
        read_start = time.time()
        cats = pickle.load(open(join(preds_save_path, "cats.pkl"), "rb"))

        df = pd.read_csv(
            join(preds_save_path, "preds.csv"),
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
            df[col] = df[col].cat.rename_categories(lambda x: list(cats[col].keys())[int(x)])
        print("reading finished: elapsed %f" % (time.time() - read_start))
        print("committing %d entries" % len(df))
        pd.to_pickle(df, join(preds_save_path, "preds.pkl"))
        print("done")
