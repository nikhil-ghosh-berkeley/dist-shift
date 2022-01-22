#!/bin/bash
for i in `seq 1 1 5`; do
/n/home13/gkaplun/.conda/envs/torch/bin/python /n/home13/gkaplun/dist-shift/run.py -m \
    save_logits=True \
    pred_save_loc='cifar10_moreparams' \
    eval_last_epoch_only=False \
    logger.project='dist-shift-cifar10' \
    datamodule.n=10000 \
    model.arch=Resnet18Wide \
    trainer.max_epochs=30 \
    trainer.val_check_interval=0.5 \
    datamodule.use_aug=True \
    datamodule.train_name="CIFAR10_train" \
    datamodule.val_names="[CIFAR10_test, CIFAR10_test, CIFAR10.1_test]" \
    logger.tags='[cifar10, 10k_samples, resnet18Wide]' \
    model.num_classes=10
done