#!/bin/bash
NUM_RUNS=4
GPU_IDS=( 0 1 2 3 ) 
NUM_GPUS=${#GPU_IDS[@]}
counter=0

for i in {1..50}
do
    gpu_idx=$((counter % $NUM_GPUS))
    gpu_id=${GPU_IDS[$gpu_idx]}

    CUDA_VISIBLE_DEVICES=${gpu_id} python run.py -m \
    save_logits=True \
    pred_save_loc='early_stopping_logits' \
    eval_last_epoch_only=False \
    datamodule.n=10000 \
    model.arch=Resnet18,Densenet121 \
    trainer.max_epochs=30 \
    trainer.val_check_interval=0.5 \
    datamodule.use_aug=True \
    logger.tags='[early_stopping_logits]' &

    sleep 2 

    counter=$((counter+1))
	 if ! ((counter % NUM_RUNS)); then
		  wait
	 fi
done