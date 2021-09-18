#!/bin/bash
python run.py -m \
pred_save_loc=predictions_early_stopping \
eval_last_epoch_only=False \
datamodule.n=10000 \
model.arch=Resnet18,Densenet121 \
trainer.max_epochs=30 \
trainer.val_check_interval=0.5 \
datamodule.use_aug=True \
logger.tags='[early_stopping]'