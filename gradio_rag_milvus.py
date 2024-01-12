from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import openai
import getpass
import gradio as gr
import os 
import random 
import time 
import numpy as np
import json
import random
from tqdm import tqdm

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
from peft import PeftModel, PeftConfig



###############################
##### Setting for the Milvus
###############################
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

    global output_results, collection, rerank_db
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
    output_results = {
        'type': 'retrieved', 
        'search_time': round(search_time,5),
        'topk': topk,
        'results': []}

    for rank, info in enumerate(search_result[0]):
        output_results['results'].append({
            'rank': rank+1,
            # 'doc_info': info.entity
            'doc_info': info.to_dict()
        })

    return output_results


def search(mode, topk, search_query):
    if 'mContriever' in mode:
        return semantic_search(topk, search_query)


def predict(message, history):
    history_langchain_format = []
    history_langchain_format.append(SystemMessage(content="""마지막 질문에 답하기 위해서 다음 문맥을 사용하세요.
답을 모르면 모른다고 말하고 답을 만들어내려고 하지 마세요.
답변은 최대한 간결하게 유지하세요.
"""))

    print("history:",history)
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    print("history_langchain_format:",history_langchain_format)

    gpt_response = llm(history_langchain_format)
    return gpt_response.content


def bot(history):
    response = "**That's cool!**"
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history


###############################
##### Setting for the options
###############################
os.environ["OPENAI_API_KEY"] = getpass.getpass() # Replace with your key
llm = ChatOpenAI(temperature=1.0, model='gpt-3.5-turbo-0613')

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

###############################
##### Options for UI (gradio)
# https://www.gradio.app/
###############################
with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Smart Patent Search with LLM Agent
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

            # gr.ChatInterface(predict) 

            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.ClearButton([msg, chatbot])

            global search_query
            search_query=""
            
    
            def respond(message, chat_history):

                retrieved_results = [
                    f"제목: {doc['doc_info']['entity']['title']}, 요약문: {doc['doc_info']['entity']['summary']}"[:500] \
                        for doc in output_results['results']][:5] 

                non_truncated_retrieved_results = [
                    f"제목: {doc['doc_info']['entity']['title']}, 요약문: {doc['doc_info']['entity']['summary']}" \
                        for doc in output_results['results']][:5] 

                print("non_truncated_retrieved_results:", [len(pair) for pair in non_truncated_retrieved_results])
                
                concatentated_retrieved_results = " \n".join(retrieved_results)[:2000]

                print(f"concatentated_retrieved_results: length - {len(concatentated_retrieved_results)}\n{concatentated_retrieved_results}")
                
                print("query:",search_query)

                history_langchain_format = []
                history_langchain_format.append(SystemMessage(content=f"""마지막 질문에 답하기 위해서 다음 문맥을 사용하세요.
                답을 모르면 모른다고 말하고 답을 만들어내려고 하지 마세요.
                답변은 최대한 간결하게 유지하세요.
                # 검색 질의: {search_query}.
                # 검색 결과: {concatentated_retrieved_results}
                """))

                print("First - history_langchain_format:",history_langchain_format)

                print("history:",chat_history)
                # print("history langchain format:",history_langchain_format)
                for human, ai in chat_history:
                    history_langchain_format.append(HumanMessage(content=human))
                    history_langchain_format.append(AIMessage(content=ai))
                history_langchain_format.append(HumanMessage(content=message))
                print("history_langchain_format:",history_langchain_format)

                gpt_response = llm(history_langchain_format)
                print("gpt_response:",gpt_response)
                bot_message = gpt_response.content

                chat_history.append((message, bot_message))
                time.sleep(2)
                return "", chat_history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])

    
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

    b1.click(search, inputs=[mode, topk, query[0]], outputs=[json_result])


demo.queue().launch()
