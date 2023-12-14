# !/bin/bash
echo "Indexing Document"

##################
#### OPTIONS & ENV Variables
##################
DATA_PATH=../data/summary_origin
PATH_TO_CORPUS=$DATA_PATH/corpus.jsonl
BSIZE=64 

CKPT='facebook/mcontriever-msmarco' # bert-base-multilingual-cased

#### For indexing & prediction
OUTPUT_DIR_NAME=./index/mcontriever-msmarco_20000

##################
#### Generate Passage Embedding
###################
echo "PATH to Index: ${PATH_TO_CORPUS}"
echo "CKPT: ${CKPT}"

CUDA_VISIBLE_DEVICES=0 python generate_passage_embeddings.py \
    --model_name_or_path $CKPT \
    --output_dir ${OUTPUT_DIR_NAME}/embeddings/ \
    --passages ${PATH_TO_CORPUS} \
    --shard_id 0  \
    --num_shards 1 \
    --per_gpu_batch_size $BSIZE
