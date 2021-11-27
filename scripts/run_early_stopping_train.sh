#!/bin/bash
#SBATCH --partition=yugroup
#SBATCH -w morgoth
#SBATCH --gres=gpu:1
#SBATCH --mem=MaxMemPerNode
#SBATCH -t 1-12:00

source `which virtualenvwrapper.sh`
workon deep-learning

for i in {1..17}; do
    python run.py -m \
    val_names='[CIFAR10_subset_train]' \
    pred_save_loc='predictions_early_stopping_train' \
    eval_last_epoch_only=False \
    logger.project='dist-shift' \
    datamodule.idx_fname='subset.pkl' \
    model.arch=Resnet18,Densenet121 \
    trainer.max_epochs=30 \
    trainer.val_check_interval=0.5 \
    datamodule.use_aug=True \
    logger.tags='[early_stopping_train]'
done
