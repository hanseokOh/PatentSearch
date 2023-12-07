# !/bin/bash
echo "Contriever training!"

# arguments: contriever/src/options.py 참고
##################
##################
#### OPTIONS & ENV Variables
##################
##################
#### For pretraining 
# PATH_TO_CORPUS=./encoded-data/bert-base-multilingual-cased/summary_corpus_2/ # for pretraining 
# # PATH_TO_OUTPUT_DIR=./checkpoint/pretrained.contriever.summary2
# PATH_TO_OUTPUT_DIR=./checkpoint/pretrained.contriever.summary2.re

# MODEL_NAME='facebook/mcontriever-msmarco' # bert-base-multilingual-cased
#MODEL_NAME='checkpoint/pretrained.contriever.summary2/checkpoint/lastlog/'

#### For indexing & prediction
# INDEX_PATH=./PATH/
# PRED_PATH=NONE
# CKPT=None

##################
##################
#### Training Model
##################
##################

## For pretraining 
# CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node=4 train.py \
#         --retriever_model_id bert-base-multilingual-cased --pooling average \
#         --train_data $PATH_TO_CORPUS --loading_mode split \
#         --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 \
#         --momentum 0.999 --moco_queue 32768 --temperature 0.05 \
#         --warmup_steps 1000 --total_steps 50000 --lr 0.00005 \
#         --scheduler linear --optim adamw --per_gpu_batch_size 256 --main_port=-1 --output_dir $PATH_TO_OUTPUT_DIR


#### For finetuning
# DATA_PATH="/mnt/sda/hanseok/projects/nable_kaist/data/processed/corpus2.size20000.summary_llm_gen.1012_ver1.4"
# # PATH_TO_TRAIN_DATA="$DATA_PATH/train.data.for_contriever.jsonl"
# PATH_TO_TRAIN_DATA="$DATA_PATH/train.w_negative.data.for_contriever.jsonl"
# PATH_TO_DEV_DATA="$DATA_PATH/dev.data.for_contriever.jsonl"
# PATH_TO_TEST_DATA="$DATA_PATH/test.data.for_contriever.jsonl"
# PATH_TO_PRED_DATA=None
# #PATH_TO_OUTPUT_DIR=./checkpoint/finetuned.contriever.summary2
# # PATH_TO_OUTPUT_DIR=./checkpoint/finetuned.contriever.summary2.ver2
# #PATH_TO_OUTPUT_DIR=./checkpoint/finetuned.contriever.summary2.from_pretrained
# # PATH_TO_OUTPUT_DIR=./checkpoint/finetuned.contriever.summary2.re3
# # PATH_TO_OUTPUT_DIR=./checkpoint/finetuned.contriever.summary2.re3.w_negative
# PATH_TO_OUTPUT_DIR=./checkpoint/finetuned.mbert-base.summary2.w_negative
# BSIZE=64 # origin:64

# # MODEL_NAME='facebook/mcontriever-msmarco' # bert-base-multilingual-cased
# MODEL_NAME='bert-base-multilingual-cased'



# SAVE_FREQ=300
# TOTAL_STEP=4000
# WARMUP_STEP=100

### For finetuning
# python finetuning.py \
# CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node=4 finetuning.py \
#         --retriever_model_id $MODEL_NAME \
#         --pooling average \
#         --train_data $PATH_TO_TRAIN_DATA \
#         --eval_data $PATH_TO_DEV_DATA \
#         --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 \
#         --momentum 0.999 --moco_queue 32768 --temperature 0.05 \
#         --warmup_steps $WARMUP_STEP --total_steps $TOTAL_STEP --lr 0.00005 \
# 	--eval_freq $SAVE_FREQ --save_freq $SAVE_FREQ \
#         --scheduler linear --optim adamw --per_gpu_batch_size $BSIZE --main_port=-1 \
#         --output_dir $PATH_TO_OUTPUT_DIR \
#         --model_path=$MODEL_NAME \
#         --negative_ctxs=0


#### with negative
# BSIZE=16 # origin:64
# CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node=4 finetuning.py \
#         --retriever_model_id $MODEL_NAME \
#         --pooling average \
#         --train_data $PATH_TO_TRAIN_DATA \
#         --eval_data $PATH_TO_DEV_DATA \
#         --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 \
#         --momentum 0.999 --temperature 0.05 \
#         --warmup_steps $WARMUP_STEP --total_steps $TOTAL_STEP --lr 0.00005 \
# 	--eval_freq $SAVE_FREQ --save_freq $SAVE_FREQ \
#         --scheduler linear --optim adamw --per_gpu_batch_size $BSIZE --main_port=-1 \
#         --output_dir $PATH_TO_OUTPUT_DIR \
#         --model_path=$MODEL_NAME \
#         --negative_ctxs=5



# PATH_TO_OUTPUT_DIR=./checkpoint/finetuned.contriever.summary2.from_pretrained.re
# MODEL_NAME='checkpoint/pretrained.contriever.summary2.re/checkpoint/lastlog/'

# CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node=2 finetuning.py \
#         --retriever_model_id $MODEL_NAME \
#         --pooling average \
#         --train_data $PATH_TO_TRAIN_DATA \
#         --eval_data $PATH_TO_DEV_DATA \
#         --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 \
#         --momentum 0.999 --moco_queue 32768 --temperature 0.05 \
#         --warmup_steps 1000 --total_steps 50000 --lr 0.00005 \
# 	--eval_steps 2000 --save_steps 2000 \
#         --scheduler linear --optim adamw --per_gpu_batch_size $BSIZE --main_port=-1 \
#         --output_dir $PATH_TO_OUTPUT_DIR \
#         --model_path=$MODEL_NAME \
#         --negative_ctxs=0



#########
### Using query_origin dataset 
#########
# MODEL_NAME='facebook/mcontriever-msmarco' # bert-base-multilingual-cased
# DATA_PATH="/mnt/sda/hanseok/projects/nable_kaist/data/processed/summary_origin2"
# EVAL_DATA_PATH="/mnt/sda/hanseok/projects/nable_kaist/data/processed/corpus2.size20000.summary_llm_gen.1012_ver1.4"
# PATH_TO_TRAIN_DATA="$DATA_PATH/train.data.for_contriever.jsonl"
# # PATH_TO_TRAIN_DATA="$DATA_PATH/train.w_negative.data.for_contriever.jsonl"
# PATH_TO_DEV_DATA="$EVAL_DATA_PATH/dev.data.for_contriever.jsonl"
# # PATH_TO_TEST_DATA="$EVAL_DATA_PATH/test.data.for_contriever.jsonl"
# PATH_TO_PRED_DATA=None

# PATH_TO_OUTPUT_DIR=./checkpoint/finetuned.contriever.summary2.query_origin
# BSIZE=64 # origin:64

# MODEL_NAME='facebook/mcontriever-msmarco' # bert-base-multilingual-cased
# # MODEL_NAME='bert-base-multilingual-cased'


# SAVE_FREQ=300
# TOTAL_STEP=4000
# WARMUP_STEP=100

# ### For finetuning
# CUDA_VISIBLE_DEVICES='1,2,3' python -m torch.distributed.launch --nproc_per_node=3 finetuning.py \
#         --retriever_model_id $MODEL_NAME \
#         --pooling average \
#         --train_data $PATH_TO_TRAIN_DATA \
#         --eval_data $PATH_TO_DEV_DATA \
#         --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 \
#         --momentum 0.999 --moco_queue 32768 --temperature 0.05 \
#         --warmup_steps $WARMUP_STEP --total_steps $TOTAL_STEP --lr 0.00005 \
# 	--eval_freq $SAVE_FREQ --save_freq $SAVE_FREQ \
#         --scheduler linear --optim adamw --per_gpu_batch_size $BSIZE --main_port=-1 \
#         --output_dir $PATH_TO_OUTPUT_DIR \
#         --model_path=$MODEL_NAME \
#         --negative_ctxs=0

#### with negative
# DATA_PATH="/mnt/sda/hanseok/projects/nable_kaist/data/processed/summary_origin2"
# EVAL_DATA_PATH="/mnt/sda/hanseok/projects/nable_kaist/data/processed/corpus2.size20000.summary_llm_gen.1012_ver1.4"
# PATH_TO_TRAIN_DATA="$DATA_PATH/train.w_negative.data.for_contriever.jsonl"
# PATH_TO_DEV_DATA="$EVAL_DATA_PATH/dev.data.for_contriever.jsonl"
# # PATH_TO_TEST_DATA="$EVAL_DATA_PATH/test.data.for_contriever.jsonl"
# PATH_TO_PRED_DATA=None

# PATH_TO_OUTPUT_DIR=./checkpoint/finetuned.contriever.summary2.query_origin.w_negative
# MODEL_NAME='facebook/mcontriever-msmarco' # bert-base-multilingual-cased
# # MODEL_NAME='bert-base-multilingual-cased'

# SAVE_FREQ=300
# TOTAL_STEP=4000
# WARMUP_STEP=100
# BSIZE=16 # origin:64
# CUDA_VISIBLE_DEVICES='1,2,3' python -m torch.distributed.launch --nproc_per_node=3 finetuning.py \
#         --retriever_model_id $MODEL_NAME \
#         --pooling average \
#         --train_data $PATH_TO_TRAIN_DATA \
#         --eval_data $PATH_TO_DEV_DATA \
#         --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 \
#         --momentum 0.999 --temperature 0.05 \
#         --warmup_steps $WARMUP_STEP --total_steps $TOTAL_STEP --lr 0.00005 \
# 	--eval_freq $SAVE_FREQ --save_freq $SAVE_FREQ \
#         --scheduler linear --optim adamw --per_gpu_batch_size $BSIZE --main_port=-1 \
#         --output_dir $PATH_TO_OUTPUT_DIR \
#         --model_path=$MODEL_NAME \
#         --negative_ctxs=5


#######
### Merge train set 
#######
DATA_PATH="/mnt/sda/hanseok/projects/nable_kaist/data/processed/summary_origin2"
EVAL_DATA_PATH="/mnt/sda/hanseok/projects/nable_kaist/data/processed/corpus2.size20000.summary_llm_gen.1012_ver1.4"
PATH_TO_TRAIN_DATA_1="$DATA_PATH/train.w_negative.data.for_contriever.jsonl"
PATH_TO_TRAIN_DATA_2="$EVAL_DATA_PATH/train.w_negative.data.for_contriever.jsonl"
# PATH_TO_DEV_DATA="$EVAL_DATA_PATH/dev.data.for_contriever.jsonl"
PATH_TO_DEV_DATA=$PATH_TO_TRAIN_DATA_1 #"$EVAL_DATA_PATH/dev.data.for_contriever.jsonl"
# PATH_TO_TEST_DATA="$EVAL_DATA_PATH/test.data.for_contriever.jsonl"
PATH_TO_OUTPUT_DIR=./checkpoint/finetuned.contriever.summary2.w_negative.merge.query_origin.llm_gen.debug
MODEL_NAME='facebook/mcontriever-msmarco' # bert-base-multilingual-cased
# MODEL_NAME='bert-base-multilingual-cased'



SAVE_FREQ=1000
TOTAL_STEP=8000
WARMUP_STEP=1000


BSIZE=24 # origin:64
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node=4 finetuning.py \
        --retriever_model_id $MODEL_NAME \
        --pooling average \
        --train_data $PATH_TO_TRAIN_DATA_1 $PATH_TO_TRAIN_DATA_2 \
        --eval_data $PATH_TO_DEV_DATA \
        --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 \
        --momentum 0.999 --temperature 0.05 \
        --warmup_steps $WARMUP_STEP --total_steps $TOTAL_STEP --lr 0.00005 \
	--eval_freq $SAVE_FREQ --save_freq $SAVE_FREQ \
        --scheduler linear --optim adamw --per_gpu_batch_size $BSIZE --main_port=-1 \
        --output_dir $PATH_TO_OUTPUT_DIR \
        --model_path=$MODEL_NAME \
        --negative_ctxs=5

