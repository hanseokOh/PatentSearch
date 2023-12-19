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
# CKPT='checkpoint/finetuned.contriever.summary2.w_negative.merge.query_origin.llm_gen_16000.fix/checkpoint/step-16000/' 

##################
#### Generate Passage Embedding
###################
echo "PATH to Index: ${PATH_TO_CORPUS}"
echo "CKPT: ${CKPT}"
   
# Build new index 
CUDA_VISIBLE_DEVICES=0 python milvus_indexing.py \
    --model_name_or_path $CKPT \
    --passages ${PATH_TO_CORPUS1} \
    --per_gpu_batch_size $BSIZE 

# When updating index  
CUDA_VISIBLE_DEVICES=0 python milvus_indexing.py \
    --model_name_or_path $CKPT \
    --passages ${PATH_TO_CORPUS2} \
    --per_gpu_batch_size $BSIZE \
    --append_mode