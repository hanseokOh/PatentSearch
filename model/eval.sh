# !/bin/bash
echo "Model Evaluation!"

# arguments: contriever/src/options.py 참고
##################
#### OPTIONS & ENV Variables
##################
# DATA_PATH="/mnt/sda/hanseok/projects/nable_kaist/data/processed/corpus2.size20000.summary_llm_gen.1012_ver1.4/"
DATA_PATH="../data/generated_summary_ver"
PATH_TO_CORPUS=$DATA_PATH/corpus.jsonl

PATH_TO_TRAIN_DATA="$DATA_PATH/train.data.for_contriever.jsonl"
PATH_TO_DEV_DATA="$DATA_PATH/dev.data.for_contriever.jsonl"
PATH_TO_TEST_DATA="$DATA_PATH/test.data.for_contriever.jsonl"
PATH_TO_PRED_DATA=None
# PATH_TO_OUTPUT_DIR=./checkpoint/finetuned.contriever.summary2
BSIZE=64 # origin:64

CKPT='facebook/mcontriever-msmarco' # bert-base-multilingual-cased
# CKPT='checkpoint/finetuned.contriever.summary2.w_negative.merge.query_origin.llm_gen.debug/checkpoint/step-1000/' # bert-base-multilingual-cased
# CKPT='checkpoint/finetuned.contriever.summary2.w_negative.merge.query_origin.llm_gen.debug/checkpoint/step-8000/' # bert-base-multilingual-cased

#### For indexing & prediction
# OUTPUT_DIR_NAME=./hs_experiment/mcontriever-summary2

##################
# #### Retrieve Passage / prediction
# ##################
# python passage_retrieval.py \
#     --model_name_or_path $CKPT \
#     --passages $PATH_TO_CORPUS \
#     --passages_embeddings "${OUTPUT_DIR_NAME}/embeddings/passages_*" \
#     --data $PATH_TO_TEST_DATA \
#     --output_dir $PRED_PATH/pred/


# ##################
# #### Evaluation
# ##################
CUDA_VISIBLE_DEVICES=1 python eval_beir.py \
    --model_name_or_path $CKPT \
    --dataset $DATA_PATH \
    --normalize_text \
    --split train