#!/bin/bash
#######
### Merge train set 
#######
DATA_PATH="../data/summary_origin"
EVAL_DATA_PATH="../data/generated_summary_ver/"
PATH_TO_TRAIN_DATA_1="$DATA_PATH/train.w_negative.data.for_contriever.jsonl"
PATH_TO_TRAIN_DATA_2="$EVAL_DATA_PATH/train.w_negative.data.for_contriever.jsonl"
PATH_TO_DEV_DATA="$EVAL_DATA_PATH/dev.data.for_contriever.jsonl"
# PATH_TO_DEV_DATA=$PATH_TO_TRAIN_DATA_1 #"$EVAL_DATA_PATH/dev.data.for_contriever.jsonl"
# PATH_TO_TEST_DATA="$EVAL_DATA_PATH/test.data.for_contriever.jsonl"
# PATH_TO_OUTPUT_DIR=./checkpoint/finetuned.contriever.summary2.w_negative.merge.query_origin.llm_gen.debug
# PATH_TO_OUTPUT_DIR=./checkpoint/finetuned.contriever.summary2.w_negative.merge.query_origin.llm_gen
PATH_TO_OUTPUT_DIR=./checkpoint/finetuned.bert.summary2.w_negative.merge.query_origin.llm_gen
# MODEL_NAME='facebook/mcontriever-msmarco' # bert-base-multilingual-cased
MODEL_NAME='bert-base-multilingual-cased'


SAVE_FREQ=1000
TOTAL_STEP=8000
WARMUP_STEP=1000
PORT=-1
# PORT=2345


BSIZE=24 # origin:64
# CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node=4 finetuning.py \
CUDA_VISIBLE_DEVICES='1' python -m torch.distributed.launch --nproc_per_node=1 finetuning.py \
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
        --negative_ctxs=5
