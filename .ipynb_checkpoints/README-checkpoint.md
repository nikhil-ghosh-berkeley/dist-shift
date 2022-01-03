# dist-shift

To run the experiment with early stopping

```
./scripts/run_early_stopping.sh
```

To run the experiment with pretrained model

```
./scripts/pretrained.sh
```

The results will be saved to "predictions" subdirectory ```pred_save_loc``` (which can be changed in ```config.yaml```). In this directory a subdirectory "raw" contains the raw data. Each file is of the form (name, hash, epoch, step) and accumulates data from each trial.

for an example of how this data is processed see "parse_results.ipynb" 

to see how the processed data can then be plotted see "plotting.ipynb"
