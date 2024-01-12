# !/bin/bash
echo "Model Evaluation!"

####################
# For base checkpoint 
python eval_beir.py --model_name_or_path facebook/mcontriever-msmarco --dataset ../data/generated_summary_ver/ --normalize_text --qrels_file high_lexical.filtered.test

python eval_beir.py --model_name_or_path facebook/mcontriever-msmarco --dataset ../data/generated_summary_ver/ --normalize_text --qrels_file filtered.test

python eval_beir.py --model_name_or_path facebook/mcontriever-msmarco --dataset ../data/new_summary_origin_20000/ --normalize_text --qrels_file high.lexical.test

python eval_beir.py --model_name_or_path facebook/mcontriever-msmarco --dataset ../data/new_summary_origin_20000/ --normalize_text --qrels_file high.semantic.test


# For PEFT tuned checkpoint
python eval_beir.py --model_path facebook/mcontriever-msmarco --dataset ../data/generated_summary_ver/ --normalize_text --qrels_file high_lexical.filtered.test --use_peft --peft_model_path checkpoint/peft_loraR.8_loraAlpha.16_lr.5e-4/checkpoint/step-20000/

python eval_beir.py --model_path facebook/mcontriever-msmarco --dataset ../data/generated_summary_ver/ --normalize_text --qrels_file filtered.test --use_peft --peft_model_path checkpoint/peft_loraR.8_loraAlpha.16_lr.5e-4/checkpoint/step-20000/

python eval_beir.py --model_path facebook/mcontriever-msmarco --dataset ../data/new_summary_origin_20000/ --normalize_text --qrels_file high.lexical.test --use_peft --peft_model_path checkpoint/peft_loraR.8_loraAlpha.16_lr.5e-4/checkpoint/step-20000/

python eval_beir.py --model_path facebook/mcontriever-msmarco --dataset ../data/new_summary_origin_20000/ --normalize_text --qrels_file high.semantic.test --use_peft --peft_model_path checkpoint/peft_loraR.8_loraAlpha.16_lr.5e-4/checkpoint/step-20000/


# For Full finetuned checkpoint
python eval_beir.py --model_path checkpoint/full_finetune.5e-5/checkpoint/step-20000/ --dataset ../data/generated_summary_ver/ --normalize_text --qrels_file high_lexical.filtered.test

python eval_beir.py --model_path checkpoint/full_finetune.5e-5/checkpoint/step-20000/ --dataset ../data/generated_summary_ver/ --normalize_text --qrels_file filtered.test 

python eval_beir.py --model_path checkpoint/full_finetune.5e-5/checkpoint/step-20000/ --dataset ../data/new_summary_origin_20000/ --normalize_text --qrels_file high.lexical.test 

python eval_beir.py --model_path checkpoint/full_finetune.5e-5/checkpoint/step-20000/ --dataset ../data/new_summary_origin_20000/ --normalize_text --qrels_file high.semantic.test