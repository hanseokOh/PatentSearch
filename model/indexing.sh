# !/bin/bash
echo "Indexing Document"

# arguments: contriever/src/options.py 참고
##################
#### OPTIONS & ENV Variables
##################
DATA_PATH=../data/origin/
PATH_TO_CORPUS=$DATA_PATH/특허Text_20230724_1.xlsx
# PATH_TO_OUTPUT_DIR=./checkpoint/finetuned.contriever.summary2
BSIZE=64 # origin:64

CKPT='facebook/mcontriever-msmarco' # bert-base-multilingual-cased
# CKPT='checkpoint/finetuned.contriever.summary2.w_negative.merge.query_origin.llm_gen.debug/checkpoint/step-1000/' # bert-base-multilingual-cased
# CKPT='checkpoint/finetuned.contriever.summary2.w_negative.merge.query_origin.llm_gen.debug/checkpoint/step-8000/' # bert-base-multilingual-cased

#### For indexing & prediction
# OUTPUT_DIR_NAME=./index/mcontriever-msmarco
# OUTPUT_DIR_NAME=./index/tuned-1000
OUTPUT_DIR_NAME=./index/tuned-8000

##################
#### Generate Passage Embedding
###################
echo "PATH to Index: ${PATH_TO_CORPUS}"
echo "CKPT: ${CKPT}"

# CUDA_VISIBLE_DEVICES=0 python generate_passage_embeddings.py \
#     --model_name_or_path $CKPT \
#     --output_dir ${OUTPUT_DIR_NAME}/embeddings/ \
#     --passages ${PATH_TO_CORPUS} \
#     --shard_id 0  \
#     --num_shards 1 \
#     --per_gpu_batch_size $BSIZE 
    
   
CUDA_VISIBLE_DEVICES=0 python milvus_indexing.py \
    --model_name_or_path $CKPT \
    --passages ${PATH_TO_CORPUS} \
    --shard_id 0  \
    --num_shards 1 \
    --per_gpu_batch_size $BSIZE 
    #     --output_dir ${OUTPUT_DIR_NAME}/embeddings/ \
