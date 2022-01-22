#!/bin/bash
for i in `seq 1 1 5`; do
/n/home13/gkaplun/.conda/envs/torch/bin/python /n/home13/gkaplun/dist-shift/run.py -m \
    save_logits=True \
    pred_save_loc='cifar10_moreparams' \
    eval_last_epoch_only=False \
    logger.project='dist-shift-cifar10' \
    datamodule.n=5000 \
    model.arch=ClipViTB16 \
    trainer.max_epochs=5 \
    trainer.val_check_interval=0.1 \
    datamodule.use_aug=True \
    datamodule.train_name="CIFAR10_train" \
    datamodule.val_names="[CIFAR10_test, CIFAR10_test]" \
    logger.tags='[cifar10, 5k_samples, clip16]' \
    model.num_classes=10
done