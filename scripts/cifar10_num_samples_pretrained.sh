#!/bin/bash

/n/home13/gkaplun/.conda/envs/torch/bin/python /n/home13/gkaplun/dist-shift/run.py -m \
    pred_save_loc='cifar10_num_samples' \
    eval_last_epoch_only=True \
    logger.project='dist-shift-cifar10' \
    datamodule.n=2500,5000,7500,10000,12500,15000,20000,25000,30000 \
    model.arch=ClipViTB32\
    trainer.max_epochs=5 \
    trainer.val_check_interval=1. \
    datamodule.use_aug=True \
    datamodule.train_name="CIFAR10_train" \
    datamodule.val_names="[CIFAR10_test, CIFAR10_test]" \
    logger.tags='[cifar10, vary_samples]' \
    model.num_classes=10

/n/home13/gkaplun/.conda/envs/torch/bin/python /n/home13/gkaplun/dist-shift/run.py -m \
    pred_save_loc='cifar10_num_samples' \
    eval_last_epoch_only=True \
    logger.project='dist-shift-cifar10' \
    datamodule.n=2500,5000,7500,10000,12500,15000,20000,25000,30000 \
    model.arch=Resnet18,Densenet121 \
    model.pretrained=True \
    trainer.max_epochs=5 \
    trainer.val_check_interval=1. \
    datamodule.use_aug=True \
    datamodule.train_name="CIFAR10_train" \
    datamodule.val_names="[CIFAR10_test, CIFAR10_test]" \
    logger.tags='[cifar10, vary_samples]' \
    model.num_classes=10

