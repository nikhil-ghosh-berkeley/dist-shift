#!/bin/bash

for i in {1..25}
do
    python run.py \
    pred_save_loc='predictions/pretrained' \
    eval_last_epoch_only=False \
    datamodule.n=5000 \
    model.arch=Densenet121 \
    model.learning_rate=0.001 \
    model.weight_decay=0 \
    model.pretrained=True \
    trainer.max_epochs=3 \
    trainer.val_check_interval=0.1 \
    datamodule.use_aug=True \
    logger.tags='[pretrained]'
done