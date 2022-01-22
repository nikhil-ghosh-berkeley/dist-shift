#!/bin/bash
/n/home13/gkaplun/.conda/envs/torch/bin/python /n/home13/gkaplun/dist-shift/run.py -m pred_save_loc='test' \
    eval_last_epoch_only=False \
    logger.project='test' \
    datamodule.n=10000 \
    model.arch=Resnet18,Densenet121 \
    trainer.max_epochs=5 \
    trainer.val_check_interval=1. \
    datamodule.use_aug=True \
    model.pretrained=True \
    datamodule.train_name="CIFAR100_train" \
    datamodule.val_names="[CIFAR100_test,CIFAR100_test]" \
    datamodule.seed=42 \
    logger.tags='[cifar100,50k_samples]' \
    model.num_classes=100