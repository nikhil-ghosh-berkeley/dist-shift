#!/bin/bash

for i in {1..5}; do
    /n/home13/gkaplun/.conda/envs/torch/bin/python /n/home13/gkaplun/dist-shift/run.py -m \
    pred_save_loc='predictions_early' \
    eval_last_epoch_only=False \
    logger.project='dist-shift' \
    datamodule.n=10000 \
    model.arch=ClipViTB32 \
    trainer.max_epochs=30 \
    trainer.val_check_interval=0.5 \
    datamodule.use_aug=True \
    logger.tags='[early_stopping]'
done