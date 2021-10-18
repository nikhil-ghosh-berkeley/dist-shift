#!/bin/bash

for i in {1..8}
do
    python run.py -m \
    pred_save_loc='predictions_vary_n' \
    eval_last_epoch_only=True \
    datamodule.n=5000,10000,15000,20000,25000,30000,35000,40000,50000 \
    model.arch=Resnet18,Densenet121 \
    trainer.max_epochs=25 \
    datamodule.use_aug=False \
    logger.tags='[vary_n]'
done