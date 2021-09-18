# dist-shift

To run the experiment with early stopping

```
./run_early_stopping.sh
```

To run the experiment with varying n

```
./run_early_stopping.sh
```

The results will be saved to directory ```pred_save_loc``` (which can be changed in ```config.yaml```). In this directory we save 

1. preds.csv
2. hash_dict.pkl
3. cats.pkl

running ```opt_df.py``` with ```pred_save_loc``` will compress ```preds.csv``` into a pickle file ```preds.pkl```

```preds.pkl``` and ```hash_dict.pkl``` will contain all the necessary info for plots
