#!/bin/bash

/n/home13/gkaplun/.conda/envs/torch/bin/python /n/home13/gkaplun/dist-shift/run.py -m \
    pred_save_loc='subsample_clip_results_new2' \
    eval_last_epoch_only=False \
    logger.project='dist-shift' \
    datamodule.n=$1 \
    model.arch=ClipViTB32 \
    trainer.max_epochs=3 \
    trainer.val_check_interval=0.1 \
    datamodule.use_aug=True \
    datamodule.seed=42 \
    logger.tags='[early_stopping]'