# !/bin/bash
echo "Model Evaluation!"

##################
#### OPTIONS & ENV Variables
##################
DATA_PATH="../data/generated_summary_ver"
BSIZE=64 

CKPT='facebook/mcontriever-msmarco' # 
# CKPT='checkpoint/finetuned.contriever.summary2.w_negative.merge.query_origin.llm_gen_16000.fix/checkpoint/step-16000/'

echo "CKPT: $CKPT"

CUDA_VISIBLE_DEVICES=0 python eval_beir.py \
    --model_name_or_path $CKPT \
    --dataset $DATA_PATH \
    --normalize_text \
    --per_gpu_batch_size $BSIZE \
    --split test 