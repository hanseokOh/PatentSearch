import numpy as np
import json
import random
import faiss
from tqdm import tqdm
import pickle
from datasets import load_dataset
from collections import defaultdict
from sklearn.cluster import KMeans
from pyserini.search.lucene import LuceneSearcher

import glob


import gradio as gr
import torch
import time
from transformers import AutoTokenizer, AutoModel
from easydict import EasyDict

from pymilvus import (
    connections,
    utility,
    Collection,
)

######################
######################
######################
# Create a Milvus connection
def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host=_HOST, port=_PORT)
    print(f"\nList connections:")
    print(connections.list_connections())

def has_collection(name):
    return utility.has_collection(name)

# List all collections in Milvus
def list_collections():
    print("\nlist collections:")
    print(utility.list_collections())

def get_entity_num(collection):
    print("\nThe number of entity:")
    print(collection.num_entities)


def load_collection(collection):
    collection.load()

def release_collection(collection):
    collection.release()

def _search(collection, topk, search_vectors):
    # Index parameters
    _METRIC_TYPE = 'IP'
    _NPROBE = 16

    search_param = {
        "data": search_vectors,
        "anns_field": 'embeddings',
        "param": {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}},
        "limit": topk,
        # "expr": "id_field >= 0",
        # 'output_fields': ["id_field","patent_id","title","text","embeddings"]
        'output_fields': ["title","summary"]
        }

    results = collection.search(**search_param)
    return results 


def semantic_search(topk, query):
    def _mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(
            ~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(
            dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    global results, collection, rerank_db
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

        query_emb = _mean_pooling(
            outputs[0], query_tensor['attention_mask']).detach().cpu().numpy()

    search_result = _search(collection, topk, query_emb)

    search_time = time.time()-retrieval_st
    print(f"### time spent for retrieval: {search_time:.4f}")

    # corpus
    results = {
        'type': 'retrieved', 
        'search_time': round(search_time,5),
        'topk': topk,
        'results': []}

    for rank, info in enumerate(search_result[0]):
        results['results'].append({
            'rank': rank+1,
            'doc_info': info.entity
        })

    return results


def search(mode, topk, query):
    if 'mContriever' in mode:
        return semantic_search(topk, query)

args = EasyDict({
    'mode': 'mContriever-msmarco',
    'model_name_or_path': 'facebook/mcontriever-msmarco'
})

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Loading model from: {args.model_name_or_path}")
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModel.from_pretrained(args.model_name_or_path)
model = model.to(device)
model.eval()

query_encoder = model
doc_encoder = model
print('load model')

# create a connection
_HOST = '127.0.0.1'
_PORT = '19530'

# Const names
_COLLECTION_NAME = 'patent'

create_connection()
if has_collection(_COLLECTION_NAME):
    print("Collection exists!\n")
    collection = Collection(_COLLECTION_NAME)
    load_collection(collection)
    get_entity_num(collection)


with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Smart Patent Search
    """
    )
    with gr.Row():
        with gr.Column():
            mode = gr.Dropdown(
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

demo.launch()
