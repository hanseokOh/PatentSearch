# !/bin/bash
echo "Model Evaluation!"

##################
#### OPTIONS & ENV Variables
##################
DATA_PATH="../data/generated_summary_ver"
# DATA_PATH="../data/new_summary_origin_20000"
BSIZE=64 

CKPT='facebook/mcontriever-msmarco' 
# Help: Use trained ckpt => change STEP Env
# CKPT='./checkpoint/re.final.finetuned.contriever/checkpoint/step-${STEP}'

echo "CKPT: $CKPT"

CUDA_VISIBLE_DEVICES=0 python eval_beir.py \
    --model_name_or_path $CKPT \
    --dataset $DATA_PATH \
    --normalize_text \
    --per_gpu_batch_size $BSIZE \
    --split test 