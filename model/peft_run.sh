#!/bin/bash
#######
### Merge train set 
#######
DATA_PATH1="../data/new_summary_origin_20000"
DATA_PATH2="../data/generated_summary_ver"
PATH_TO_TRAIN_DATA_1="$DATA_PATH1/train.w_negative.data.for_contriever.jsonl"
PATH_TO_TRAIN_DATA_2="$DATA_PATH2/train.w_negative.data.for_contriever.jsonl"
PATH_TO_DEV_DATA="$DATA_PATH2/dev.data.for_contriever.jsonl"
MODEL_NAME='facebook/mcontriever-msmarco' 
SAVE_FREQ=1000
TOTAL_STEP=20000
WARMUP_STEP=1000


# Shared Hyperparameters 
BSIZE=24
DEVICES='2,3'

# Different Hyperparameters 
LR=0.00005
WANDB_RUN='full_finetune.5e-5'
PATH_TO_OUTPUT_DIR=./checkpoint/$WANDB_RUN

CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch --nproc_per_node=2 finetuning.py \
        --retriever_model_id $MODEL_NAME \
        --pooling average \
        --train_data $PATH_TO_TRAIN_DATA_1 $PATH_TO_TRAIN_DATA_2 \
        --eval_data $PATH_TO_DEV_DATA \
        --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 \
        --momentum 0.999 --temperature 0.05 \
        --warmup_steps $WARMUP_STEP --total_steps $TOTAL_STEP --lr $LR \
        --eval_freq $SAVE_FREQ --save_freq $SAVE_FREQ \
        --scheduler linear --optim adamw --per_gpu_batch_size $BSIZE --main_port=-1 \
        --output_dir $PATH_TO_OUTPUT_DIR \
        --model_path=$MODEL_NAME \
        --negative_ctxs=5 \
        --eval_datasets 'generated_summary_ver' 'new_summary_origin_20000' \
        --eval_datasets_dir ../data  \
        --wandb_run_name $WANDB_RUN



LR=0.00005
LORA_R=8
LORA_DROP=0.1
LORA_ALPHA=16
WANDB_RUN='peft_loraR.8_loraAlpha.16_lr.5e-5'
PATH_TO_OUTPUT_DIR=./checkpoint/$WANDB_RUN


CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch --nproc_per_node=2 finetuning.py \
        --retriever_model_id $MODEL_NAME \
        --pooling average \
        --train_data $PATH_TO_TRAIN_DATA_1 $PATH_TO_TRAIN_DATA_2 \
        --eval_data $PATH_TO_DEV_DATA \
        --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 \
        --momentum 0.999 --temperature 0.05 \
        --warmup_steps $WARMUP_STEP --total_steps $TOTAL_STEP --lr $LR \
        --eval_freq $SAVE_FREQ --save_freq $SAVE_FREQ \
        --scheduler linear --optim adamw --per_gpu_batch_size $BSIZE --main_port=-1 \
        --output_dir $PATH_TO_OUTPUT_DIR \
        --model_path=$MODEL_NAME \
        --negative_ctxs=5 \
        --eval_datasets 'generated_summary_ver' 'new_summary_origin_20000' \
        --eval_datasets_dir ../data  \
        --use_peft \
        --lora_r $LORA_R \
        --lora_dropout $LORA_DROP \
        --lora_alpha $LORA_ALPHA \
        --wandb_run_name $WANDB_RUN 


LR=0.0005
LORA_R=8
LORA_DROP=0.1
LORA_ALPHA=16
WANDB_RUN='peft_loraR.8_loraAlpha.16_lr.5e-4'
PATH_TO_OUTPUT_DIR=./checkpoint/$WANDB_RUN


CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch --nproc_per_node=2 finetuning.py \
        --retriever_model_id $MODEL_NAME \
        --pooling average \
        --train_data $PATH_TO_TRAIN_DATA_1 $PATH_TO_TRAIN_DATA_2 \
        --eval_data $PATH_TO_DEV_DATA \
        --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 \
        --momentum 0.999 --temperature 0.05 \
        --warmup_steps $WARMUP_STEP --total_steps $TOTAL_STEP --lr $LR \
        --eval_freq $SAVE_FREQ --save_freq $SAVE_FREQ \
        --scheduler linear --optim adamw --per_gpu_batch_size $BSIZE --main_port=-1 \
        --output_dir $PATH_TO_OUTPUT_DIR \
        --model_path=$MODEL_NAME \
        --negative_ctxs=5 \
        --eval_datasets 'generated_summary_ver' 'new_summary_origin_20000' \
        --eval_datasets_dir ../data  \
        --use_peft \
        --lora_r $LORA_R \
        --lora_dropout $LORA_DROP \
        --lora_alpha $LORA_ALPHA \
        --wandb_run_name $WANDB_RUN 

LR=0.005
LORA_R=8
LORA_DROP=0.1
LORA_ALPHA=16
WANDB_RUN='peft_loraR.8_loraAlpha.16_lr.5e-3'
PATH_TO_OUTPUT_DIR=./checkpoint/$WANDB_RUN

CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch --nproc_per_node=2 finetuning.py \
        --retriever_model_id $MODEL_NAME \
        --pooling average \
        --train_data $PATH_TO_TRAIN_DATA_1 $PATH_TO_TRAIN_DATA_2 \
        --eval_data $PATH_TO_DEV_DATA \
        --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 \
        --momentum 0.999 --temperature 0.05 \
        --warmup_steps $WARMUP_STEP --total_steps $TOTAL_STEP --lr $LR \
        --eval_freq $SAVE_FREQ --save_freq $SAVE_FREQ \
        --scheduler linear --optim adamw --per_gpu_batch_size $BSIZE --main_port=-1 \
        --output_dir $PATH_TO_OUTPUT_DIR \
        --model_path=$MODEL_NAME \
        --negative_ctxs=5 \
        --eval_datasets 'generated_summary_ver' 'new_summary_origin_20000' \
        --eval_datasets_dir ../data  \
        --use_peft \
        --lora_r $LORA_R \
        --lora_dropout $LORA_DROP \
        --lora_alpha $LORA_ALPHA \
        --wandb_run_name $WANDB_RUN
