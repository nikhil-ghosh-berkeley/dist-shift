#!/bin/bash

for i in {1..25}
do
    python run.py -m \
    pred_save_loc='predictions_early_stopping' \
    eval_last_epoch_only=False \
    datamodule.n=10000 \
    model.arch=Resnet18,Densenet121 \
    trainer.max_epochs=30 \
    trainer.val_check_interval=0.5 \
    datamodule.use_aug=True \
    logger.tags='[early_stopping]'
done

 python run.py -m pred_save_loc='predictions_early_stopping' eval_last_epoch_only=False datamodule.n=10000 model.arch=ClipViTB16 trainer.max_epochs=30 trainer.val_check_interval=0.5 datamodule.use_aug=True logger.tags='[early_stopping]'