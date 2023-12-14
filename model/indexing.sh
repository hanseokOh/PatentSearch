# !/bin/bash
echo "Indexing Document"

# arguments: src/options.py 
##################
#### OPTIONS & ENV Variables
##################
DATA_PATH=../data/origin
PATH_TO_CORPUS=$DATA_PATH/특허Text_20230724_1.xlsx
# PATH_TO_CORPUS=$DATA_PATH/특허Text_20230724_2.xlsx
BSIZE=64 

CKPT='facebook/mcontriever-msmarco' 

##################
#### Generate Passage Embedding
###################
echo "PATH to Index: ${PATH_TO_CORPUS}"
echo "CKPT: ${CKPT}"
   
CUDA_VISIBLE_DEVICES=0 python milvus_indexing.py \
    --model_name_or_path $CKPT \
    --passages ${PATH_TO_CORPUS} \
    --per_gpu_batch_size $BSIZE \
    --append_mode