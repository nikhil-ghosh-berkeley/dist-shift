#!/bin/bash
python run.py \
val_names='[]' \
pred_save_loc='train_imagenet32' \
logger.project='train_imagenet32' \
datamodule.name='imagenet32' \
datamodule.batch_size=256 \
model.num_classes=1000 \
model.arch=Resnet18 \
trainer.max_epochs=100 \
trainer.checkpoint_callback=True \
trainer.gpus=-1 \
datamodule.use_aug=True
