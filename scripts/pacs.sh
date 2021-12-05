#!/bin/bash

for i in {1..1}
do
    python run.py -m \
    val_names='[sketch_train]' \
    pred_save_loc='pacs' \
    eval_last_epoch_only=True \
    datamodule.n=50000 \
    model.arch=Resnet18,Densenet121 \
    trainer.max_epochs=100 \
    trainer.val_check_interval=1.0 \
    datamodule.use_aug=True \
    logger.tags='[pacs]'
done