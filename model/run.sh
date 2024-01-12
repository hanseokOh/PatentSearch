#!/bin/bash
#######
### Merge train set 
#######
DATA_PATH1="../data/new_summary_origin_20000"
DATA_PATH2="../data/generated_summary_ver"
PATH_TO_TRAIN_DATA_1="$DATA_PATH1/train.w_negative.data.for_contriever.jsonl"
PATH_TO_TRAIN_DATA_2="$DATA_PATH2/train.w_negative.data.for_contriever.jsonl"
PATH_TO_DEV_DATA="$DATA_PATH2/dev.data.for_contriever.jsonl"
# PATH_TO_OUTPUT_DIR=./checkpoint/re.final.finetuned.contriever
PATH_TO_OUTPUT_DIR=./checkpoint/re2.final.finetuned.contriever

MODEL_NAME='facebook/mcontriever-msmarco' 
SAVE_FREQ=1000
TOTAL_STEP=20000
WARMUP_STEP=1000
PORT=-1

BSIZE=24 
CUDA_VISIBLE_DEVICES='1,3' python -m torch.distributed.launch --nproc_per_node=2 finetuning.py \
        --retriever_model_id $MODEL_NAME \
        --pooling average \
        --train_data $PATH_TO_TRAIN_DATA_1 $PATH_TO_TRAIN_DATA_2 \
        --eval_data $PATH_TO_DEV_DATA \
        --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 \
        --momentum 0.999 --temperature 0.05 \
        --warmup_steps $WARMUP_STEP --total_steps $TOTAL_STEP --lr 0.00005 \
        --eval_freq $SAVE_FREQ --save_freq $SAVE_FREQ \
        --scheduler linear --optim adamw --per_gpu_batch_size $BSIZE --main_port=$PORT \
        --output_dir $PATH_TO_OUTPUT_DIR \
        --model_path=$MODEL_NAME \
        --negative_ctxs=5 \
        --eval_datasets 'generated_summary_ver' 'new_summary_origin_20000' \
        --eval_datasets_dir ../data  
