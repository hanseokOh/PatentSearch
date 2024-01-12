import gradio as gr
import torch
import numpy as np
import json
import random
import faiss
from tqdm import tqdm
import pickle
from datasets import load_dataset
from collections import defaultdict
import time
from easydict import EasyDict
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import glob
from peft import PeftModel, PeftConfig

corpus_path = 'data/summary_origin/corpus.jsonl'
# save_file = 'index/mcontriever-msmarco_20000/embeddings/passages_*'
save_file = 'model/index/peft_tuned.5e-4.mcontriever-msmarco_20000/embeddings/passages_*'

corpus = load_dataset('json', data_files={
    'corpus': corpus_path})['corpus']

print("corpus path: ", corpus_path)
print("corpus:", len(corpus))
# print("index file for BM25:", index_file)

embedding_files = glob.glob(save_file)
print("embedding_files:", embedding_files)
allids = []
allembeddings = np.array([])
for i, file_path in enumerate(embedding_files):
    print(f"Loading file {file_path}")
    with open(file_path, "rb") as fin:
        ids, embeddings = pickle.load(fin)

    allembeddings = np.vstack(
        (allembeddings, embeddings)) if allembeddings.size else embeddings
    allids.extend([int(id[1:]) for id in ids])

allids = np.array(allids)

print("all ids:", allids.shape)
print(f"all embeddings size: {allembeddings.shape}")

d = 768
index = faiss.IndexFlatIP(d)   # build the index
index = faiss.IndexIDMap2(index)
print("index.is_trained:", index.is_trained)
index.add_with_ids(allembeddings.astype('float32'), allids)


print("index.ntotal:", index.ntotal)
print("Data indexing completed.")


def semantic_search(topk, query):
    def _mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(
            ~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(
            dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    global results, re_index

    retrieval_st = time.time()
    with torch.no_grad():
        query_tensor = tokenizer(
            query,
            return_tensors="pt",
            max_length=32,
            padding=True,
            truncation=True,
        )
        query_tensor = {k: v.to(device) for k, v in query_tensor.items()}
        outputs = query_encoder(**query_tensor)

        ###### Needed for OPTION1 / OPTION3 model loading
        query_emb = _mean_pooling(
            outputs[0], query_tensor['attention_mask']).detach().cpu().numpy()
        
        ###### FOR Model loading OPTION2
        # query_emb = outputs.detach().cpu().numpy()

    D, I = index.search(query_emb, topk)

    print("### time spent for retrieval:", time.time()-retrieval_st)

    # corpus
    results = {'type': 'retrieved', 'topk': topk,
               'index': save_file, 'results': []}
    for rank, (score, idx) in enumerate(zip(D[0], I[0])):
        results['results'].append({
            'rank': rank+1,
            'score': float(score),
            'doc_info': corpus[int(idx)]
        })

    # build new index for rerank using bi-encoder
    selected_emb = np.array([])
    embs = np.array([list(index.reconstruct_n(int(doc['doc_info']['_id'][1:]), 1)[0])
                     for doc in results['results']])

    re_allids = np.array([int(doc['doc_info']['_id'][1:])
                         for doc in results['results']])
    selected_emb = np.vstack(
        (selected_emb, embs)) if selected_emb.size else embs

    print("re_allids:", re_allids.shape)
    print(f"selected_emb size: {selected_emb.shape}")

    d = 768
    re_index = faiss.IndexFlatIP(d)   # build the index
    re_index = faiss.IndexIDMap2(re_index)
    print(re_index.is_trained)

    re_index.add_with_ids(selected_emb.astype('float32'), re_allids)

    print("re_index.ntotal:", re_index.ntotal)
    print("Data indexing completed.")

    return results 

def search(mode, topk, query):
    s = time.time()
    return semantic_search(topk, query)


def rerank(query, rerank_topk):
    ############
    # Rerank with bi encoder
    ############
    rerank_st = time.time()
    with torch.no_grad():
        query_tensor = tokenizer(
            query,
            return_tensors="pt",
            max_length=32,
            padding=True,
            truncation=True,
        )
        query_tensor = {k: v.to(device) for k, v in query_tensor.items()}
        outputs = query_encoder(**query_tensor)
        query_emb = mean_pooling(
            outputs[0], query_tensor['attention_mask']).detach().cpu().numpy()


    D, I = re_index.search(query_emb, rerank_topk)
    print("### time spent for rerank:", time.time()-rerank_st)

    re_results = {'type': 'reranked', 'topk': rerank_topk,
                  'index': save_file + '.after_retrieval', 'results': []}
    for rank, (score, idx) in enumerate(zip(D[0], I[0])):
        re_results['results'].append({
            'rank': rank+1,
            'score': float(score),
            'doc_info': corpus[int(idx)]
        })

    return re_results


device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = EasyDict({
    'mode': 'mContriever-msmarco',
    'model_name_or_path': 'facebook/mcontriever-msmarco',
    'peft_model_path': 'checkpoint/peft_loraR.8_loraAlpha.16_lr.5e-4/checkpoint/step-20000/',
    'use_peft':True,
})

print(f"Loading model from: {args.model_name_or_path}")

###### OPTION1
# tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
# model = AutoModel.from_pretrained(args.model_name_or_path)

###### OPTION2
# model, tokenizer, _ = src.contriever.load_retriever(args.model_name_or_path)
# if args.use_peft:
#     peft_model_id = args.peft_model_path
#     print(f"PEFT mode activated - [PEFT ckpt] {peft_model_id} &  [Base ckpt] {args.model_name_or_path}")
#     config = PeftConfig.from_pretrained(peft_model_id)
#     model = PeftModel.from_pretrained(model, peft_model_id)
# else:
#     print(f"Model loaded from {args.model_name_or_path}.", flush=True)

###### OPTION3
def get_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('facebook/mcontriever-msmarco')
model = get_model('hanseokOh/smartPatent-mContriever-lora')

model = model.to(device)
model.eval()

query_encoder = model
doc_encoder = model

print('load model')

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Smart Patent Search
    """
    )
    with gr.Row():
        with gr.Column():
            mode = gr.Dropdown(
                # ["bm25", "mContriever-msmarco"],
                ["mContriever-msmarco"],
                value="mContriever-msmarco",
                label="mode",
                info="select retrieval model")

            topk = gr.Slider(1, 100, step=1, value=10,
                             label='TopK', show_label=True)
            query = gr.Textbox(
                placeholder="Enter query here...", label="Query"),

            b1 = gr.Button("Run the query")
            b3 = gr.Button("Run the further query")

            gr.Examples(
                examples=[
                    [20, '작업장 위험 알림시스템'],
                    [20, '객체를 인식하고 충돌 가능성을 예측해주는 작업장 위험 알림 시스템'],
                    [20, '충돌사고 예방을 위한 작업장 위험 알림 시스템'],
                    [20, '채널 대역폭 400MHz를 사용하는 하향링크 신호 수신 방법에 대한 특허'],
                    [20, '다중 객체의 빅데이터 획득을 위한 객체 분류 방법을 제공하는 장치'],
                    [20, '밀리미터파/테라헤르츠 주파수 범위에서 작동하는 사이드링크 통신 장치를 찾아보세요.'],
                    [20, '사물인터넷 네트워크 과부하를 방지하는 객체감지 가능한 사물 AI 시스템'],
                    [20, '단말의 송신 규격'],
                    [20, '단말의 수신 규격'],
                    [20, '객체 분류 및 다중 객체 빅데이터 획득 관리 장치'],
                    [20,
                        '작업장에서 사용되는 관찰 기둥 상단에 설치된 영상 모듈과 감지 모듈을 활용하여 객체를 식별하고 충돌 가능성을 예측하는 시스템에 대한 특허를 찾아주세요.'],
                    [20,
                        '작업자의 안전을 위해 작업 공간에서 객체를 감지하고 경고 메시지를 전송하는 방법과 관련된 특허를 검색해주세요.'],
                    [20,
                        '파워클래스3 유저 장비를 위한 전송 전력 결정 방법과 이에 기초한 상향링크 신호 전송 방법에 대한 특허를 검색해주세요.'],
                    [20,
                        '유효 등방성 복사 전력과 구형 적용 범위를 활용하여 전송 전력을 결정하는 특허를 찾아주세요.'],
                ],
                inputs=[topk, query[0]]
            )
        with gr.Column():
            with gr.Accordion("See Details for retrieved output"):
                json_result = gr.JSON(label="json output")

    print("## DEBUG:", args, mode, topk, query[0])
    b1.click(search, inputs=[mode, topk, query[0]], outputs=[json_result])
    b3.click(rerank, inputs=[query[0], topk],
             outputs=[json_result])

demo.launch()
