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

from sklearn.cluster import KMeans
from pyserini.search.lucene import LuceneSearcher
import openai

# import src.contriever
import glob
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)


# summary_corpus 20000
connections.connect("default", host="localhost", port="19530")
collection_name= "patent_search"
collection = Collection(collection_name)      # Get an existing collection.
collection.load()
print("Collection:",collection)
print("Data indexing completed.")


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(
        ~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(
        dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def semantic_search(topk, query):
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

        query_emb = mean_pooling(
            outputs[0], query_tensor['attention_mask']).detach().cpu().numpy()

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }

    search_result = collection.search(
        query_emb, 
        "embeddings", 
        search_params, 
        limit=topk, 
        output_fields=["ids","patent_id","title","text","embeddings"]
    )


    print("### time spent for retrieval:", time.time()-retrieval_st)

    # corpus
    results = {'type': 'retrieved', 'topk': topk,'results': []}
    
    for rank, info in enumerate(search_result[0]):
        results['results'].append({
            'rank': rank+1,
#             'score': float(info.distance),
            'doc_info': info.entity
        })

    # build new index for rerank using bi-encoder
    selected_results= [doc['doc_info'].to_dict()['entity'] for doc in results['results']]

    print("selected_results:",len(selected_results))
        
    fields = [
        FieldSchema(name="ids", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="patent_id", dtype=DataType.VARCHAR,max_length=50),
        FieldSchema(name="title", dtype=DataType.VARCHAR,max_length=1000),
        FieldSchema(name="text", dtype=DataType.VARCHAR,max_length=10000),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]

    schema = CollectionSchema(fields, "Temporary collection for rerank")
    
    if utility.has_collection("re_patent_search"):
        print("Drop existing collection")
        utility.drop_collection("re_patent_search")
        
    rerank_db = Collection("re_patent_search", schema)
    print("New collection added")

    #     dict_keys(['ids', 'patent_id', 'title', 'text', 'embeddings'])
    entities = [
        [pair['ids'] for pair in selected_results], # field pk
        [pair['patent_id'] for pair in selected_results], # field patent_id
        [pair['title'] for pair in selected_results], # field title
        [pair['text'] for pair in selected_results], # field text
        [pair['embeddings'] for pair in selected_results] # field embeddings
    ]

    insert_result = rerank_db.insert(entities)
    rerank_db.flush()
    
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }

    rerank_db.create_index("embeddings", index)
    rerank_db.load()
    print("Data indexing completed.")

    return results


def keyword_search(topk, query):
    searcher = LuceneSearcher(index_file)
    hits = searcher.search(q=query, k=topk)
    print("hits:", hits)
    results = {'topk': topk, 'index': index_file, 'results': []}

    for rank in range(len(hits)):
        print(f'{rank+1:2} {hits[rank].docid:4} {hits[rank].score:.5f}')

        docid = hits[rank].docid
        score = hits[rank].score
        idx = int(docid[1:])

        results['results'].append({
            'rank': rank+1,
            'score': float(score),
            'doc_info': corpus[int(idx)]
        })
    return results


def search(mode, topk, query):
    s = time.time()
    if 'mContriever' in mode:
        return semantic_search(topk, query)
    else:
        return keyword_search(topk, query)


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
        
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }

    search_result = rerank_db.search(
        query_emb, 
        "embeddings", 
        search_params, 
        limit=rerank_topk, 
        output_fields=["ids","patent_id","title","text","embeddings"]
    )

    print("### time spent for rerank:", time.time()-rerank_st)

    # corpus
    re_results = {'type': 'reranked', 'topk': rerank_topk,'results': []}
    
    for rank, info in enumerate(search_result[0]):
        re_results['results'].append({
            'rank': rank+1,
            'doc_info': info.entity
        })


    return re_results


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

with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Patent Retrieval System
    """
    )
    with gr.Row():
        with gr.Column():
            mode = gr.Dropdown(
                ["bm25", "mContriever-msmarco"],
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
