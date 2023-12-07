# !/bin/bash
echo "Model Evaluation!"

# arguments: contriever/src/options.py 참고
##################
##################
#### OPTIONS & ENV Variables
##################
##################
DATA_PATH="/mnt/sda/hanseok/projects/nable_kaist/data/processed/corpus2.size20000.summary_llm_gen.1012_ver1.4/"
PATH_TO_CORPUS=$DATA_PATH/corpus.jsonl

PATH_TO_TRAIN_DATA="$DATA_PATH/train.data.for_contriever.jsonl"
PATH_TO_DEV_DATA="$DATA_PATH/dev.data.for_contriever.jsonl"
PATH_TO_TEST_DATA="$DATA_PATH/test.data.for_contriever.jsonl"
PATH_TO_PRED_DATA=None
# PATH_TO_OUTPUT_DIR=./checkpoint/finetuned.contriever.summary2
BSIZE=64 # origin:64

CKPT='facebook/mcontriever-msmarco' # bert-base-multilingual-cased
# CKPT=./checkpoint/finetuned.contriever.summary2/checkpoint/step-300000 # bert-base-multilingual-cased
# CKPT=./checkpoint/finetuned.contriever.summary2/checkpoint/step-50000 # bert-base-multilingual-cased
# CKPT=./checkpoint/finetuned.contriever.summary2.from_pretrained/checkpoint/step-50000/

#### For indexing & prediction
# OUTPUT_DIR_NAME=./hs_experiment/finetuned.contriever.summary2.step-300000
# OUTPUT_DIR_NAME=./hs_experiment/finetuned.contriever.summary2.step-50000
# OUTPUT_DIR_NAME=./hs_experiment/finetuned.contriever.summary2.from_pretrained.step-50000
OUTPUT_DIR_NAME=./hs_experiment/mcontriever-summary2
# PRED_PATH=NONE

##################
#### Generate Passage Embedding
# ##################
# BSIZE=128
# echo "indexing Data: ${DATA}"
# for i in {0..3}; do
# 	export CUDA_VISIBLE_DEVICES=${i}
# 	nohup python generate_passage_embeddings.py --model_name_or_path $CKPT --output_dir ${OUTPUT_DIR_NAME}/embeddings/ \
# 	--passages $PATH_TO_CORPUS --shard_id ${i}  --num_shards 4 \
#     --per_gpu_batch_size $BSIZE > ./log/nohup.log.${i} 2>&1 &
# done

# CUDA_VISIBLE_DEVICES=0 nohup python generate_passage_embeddings.py --model_name_or_path $CKPT --output_dir ${OUTPUT_DIR_NAME}/embeddings/${DATA} \
#       --passages ../../../data/corss_task_cross_domain_final/${DATA}/corpus.jsonl --shard_id 0  --num_shards 8 --per_gpu_batch_size $BSIZE > ./log/nohup.log.0 2>&1 &

# python generate_passage_embeddings.py --model_name_or_path $CKPT --output_dir ${NEW_OUTPUT_DIR_NAME}/embeddings/${DATA} \
#       --passages ../../../data/corss_task_cross_domain_final/${DATA}/corpus.jsonl --shard_id 7  --num_shards 6


##################
# #### Retrieve Passage / prediction
# ##################
python passage_retrieval.py \
    --model_name_or_path $CKPT \
    --passages $PATH_TO_CORPUS \
    --passages_embeddings "${OUTPUT_DIR_NAME}/embeddings/passages_*" \
    --data $PATH_TO_TEST_DATA \
    --output_dir $PRED_PATH/pred/


# ##################
# #### Evaluation
# ##################
python eval_beir.py \
    --model_name_or_path $CKPT \
    --dataset $DATA_PATH \
    --normalize_text


# python eval_beir.py \
#         --model_name_or_path checkpoint/summary_10000_256bsz_4gpu/checkpoint/lastlog/ \
#         --dataset ../../data/processed/summary_llm_gen \
#         --normalize_text
