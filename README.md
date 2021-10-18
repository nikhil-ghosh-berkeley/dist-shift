# dist-shift

To run the experiment with early stopping

```
./scripts/run_early_stopping.sh
```

To run the experiment with varying n

```
./scripts/run_early_stopping.sh
```

The results will be saved to directory ```pred_save_loc``` (which can be changed in ```config.yaml```). In this directory the following are saved

1. preds.csv
2. hash_dict.pkl
3. cats.pkl

running ```commit_csv.py``` with ```pred_save_loc``` will add the data from ```preds.csv``` to a pickle file ```preds.pkl```

To make the plots, first run ```grouping.py``` to group the data entries of ```preds.pkl``` by model, data index, and label which is saved in ```group_by_model.pkl```, ```point_agg.pkl```, and ```label_agg.pkl``` respectively.

Then run ```process_points.py``` which processes the groupings to get scores for each point and aggregates all the information for the plots in ```proc_labels.pkl``` and ```proc_points.pkl```

Using ```proc_labels.pkl``` and ```proc_points.pkl```, ```plots.py``` will create the required plots.
