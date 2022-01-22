#!/bin/bash
for i in 1 2 3 4 5; do
/n/home13/gkaplun/.conda/envs/torch/bin/python /n/home13/gkaplun/dist-shift/run.py -m \
    pred_save_loc='cifar100_clip_new' \
    eval_last_epoch_only=False \
    logger.project='dist-shift' \
    datamodule.n=20000 \
    model.arch=ClipViTB16,ClipViTB32\
    trainer.max_epochs=5 \
    trainer.val_check_interval=0.1 \
    datamodule.use_aug=True \
    datamodule.train_name="CIFAR100_train" \
    datamodule.val_names="[CIFAR100_test, CIFAR100_test]" \
    datamodule.seed=42 \
    logger.tags='[cifar100, 20k_samples]' \
    model.num_classes=100
done
    
