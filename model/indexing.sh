# !/bin/bash
echo "Indexing Document"

##################
#### OPTIONS & ENV Variables
##################
DATA_PATH=../data/origin
PATH_TO_CORPUS1=$DATA_PATH/특허Text_20230724_1.xlsx
PATH_TO_CORPUS2=$DATA_PATH/특허Text_20230724_2.xlsx
BSIZE=64 
CKPT='facebook/mcontriever-msmarco' 
DEVICE=1

##################
#### Generate Passage Embedding
###################
# echo "PATH to Index: ${PATH_TO_CORPUS}"
# echo "CKPT: ${CKPT}"
   
# # Build new index 
# CUDA_VISIBLE_DEVICES=0 python milvus_indexing.py \
#     --model_name_or_path $CKPT \
#     --passages ${PATH_TO_CORPUS1} \
#     --per_gpu_batch_size $BSIZE \

# # When updating index  
# CUDA_VISIBLE_DEVICES=0 python milvus_indexing.py \
#     --model_name_or_path $CKPT \
#     --passages ${PATH_TO_CORPUS2} \
#     --per_gpu_batch_size $BSIZE \
#     --append_mode


##################
#### Generate Passage Embedding using PEFT tuned ckpt
###################
PEFT_CKPT='checkpoint/peft_loraR.8_loraAlpha.16_lr.5e-4/checkpoint/step-20000/'

OUTPUT_DIR_NAME=./index/peft_tuned.5e-4.mcontriever-msmarco_20000

# Build new index 
CUDA_VISIBLE_DEVICES=$DEVICE python milvus_indexing.py \
    --model_name_or_path $CKPT \
    --passages ${PATH_TO_CORPUS1} \
    --per_gpu_batch_size $BSIZE \
    --use_peft \
    --peft_model_path $PEFT_CKPT

# When updating index  
CUDA_VISIBLE_DEVICES=$DEVICE python milvus_indexing.py \
    --model_name_or_path $CKPT \
    --passages ${PATH_TO_CORPUS2} \
    --per_gpu_batch_size $BSIZE \
    --append_mode \
    --use_peft \
    --peft_model_path $PEFT_CKPT
