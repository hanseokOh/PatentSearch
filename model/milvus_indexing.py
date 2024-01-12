import random
import os
import argparse
import csv
import logging
import pickle

import numpy as np
import torch
from pymilvus import (
    connections,
    utility,
    FieldSchema,CollectionSchema,DataType,
    Collection,
)

import src.slurm
import src.contriever
import src.utils
import src.data
import src.normalize_text

from peft import PeftModel, PeftConfig


_HOST = '127.0.0.1'
_PORT = '19530'

# Const names
_COLLECTION_NAME = 'patent'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'embeddings'

# Index parameters
_METRIC_TYPE = 'IP'
_INDEX_TYPE = 'IVF_FLAT'
_NLIST = 1024
_NPROBE = 16
_TOPK = 5
MAX_LENGTH=65535

# Create a Milvus connection
def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host=_HOST, port=_PORT)
    print(f"\nList connections:")
    print(connections.list_connections())


def create_collection(name, id_field, vector_field):
    #########
    fields = [
        FieldSchema(name=id_field, dtype=DataType.VARCHAR, max_length=50, is_primary=True, auto_id=False),
        FieldSchema(name="patent_id", dtype=DataType.VARCHAR,max_length=50),
        FieldSchema(name="title", dtype=DataType.VARCHAR,max_length=1000),
        FieldSchema(name="summary", dtype=DataType.VARCHAR,max_length=MAX_LENGTH),
        FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, dim=768)
    ]

    schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")        
    collection = Collection(name=name, data=None,schema =schema)
    print("\ncollection created:", name)
    
    return collection


def has_collection(name):
    return utility.has_collection(name)


# Drop a collection in Milvus
def drop_collection(name):
    collection = Collection(name)
    collection.drop()
    print("\nDrop collection: {}".format(name))


# List all collections in Milvus
def list_collections():
    print("\nlist collections:")
    print(utility.list_collections())


def insert(collection, entities):
    collection.insert(entities)
    return entities[-1]


def get_entity_num(collection):
    print("\nThe number of entity:")
    print(collection.num_entities)


def create_index(collection, filed_name):
    index_param = {
        "index_type": _INDEX_TYPE,
        "params": {"nlist": _NLIST},
        "metric_type": _METRIC_TYPE}
    collection.create_index(filed_name, index_param)
    print("\nCreated index:\n{}".format(collection.index().params))


def drop_index(collection):
    collection.drop_index()
    print("\nDrop index sucessfully")


def load_collection(collection):
    collection.load()


def release_collection(collection):
    collection.release()


# def search(collection, vector_field, id_field, search_vectors):
def search(collection, vector_field, search_vectors):
    search_param = {
        "data": search_vectors,
        "anns_field": vector_field,
        "param": {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}},
        "limit": _TOPK,
        # "expr": "id_field >= 0"
        }
    results = collection.search(**search_param)
    for i, result in enumerate(results):
        print("\nSearch result for {}th vector: ".format(i))
        for j, res in enumerate(result):
            print("Top {}: {}".format(j, res))


def set_properties(collection):
    collection.set_properties(properties={"collection.ttl.seconds": 1800})



def embed_passages(args, passages, model, tokenizer):
    total = 0
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []
    with torch.no_grad():
        for k, p in enumerate(passages):
            # batch_ids.append(p["id"])
            batch_ids.append(p["id"] if p.get('id') else p['_id'])
            if args.no_title or not "title" in p:
                text = p["summary"]
            else:
                text = p["title"] + " " + p["summary"]
            if args.lowercase:
                text = text.lower()
            if args.normalize_text:
                text = src.normalize_text.normalize(text)
            batch_text.append(text)

            if len(batch_text) == args.per_gpu_batch_size or k == len(passages) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_text,
                    return_tensors="pt",
                    max_length=args.passage_maxlength,
                    padding=True,
                    truncation=True,
                )

                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                embeddings = model(**encoded_batch)

                embeddings = embeddings.cpu()
                total += len(batch_ids)
                allids.extend(batch_ids)
                allembeddings.append(embeddings)

                batch_text = []
                batch_ids = []
                if k % 100000 == 0 and k > 0:
                    print(f"Encoded passages {total}")

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    return allids, allembeddings


def main(args):
    # create a connection
    model, tokenizer, _ = src.contriever.load_retriever(args.model_name_or_path)

    ## add    
    if args.use_peft:
        peft_model_id = args.peft_model_path
        print(f"PEFT mode activated - [PEFT ckpt] {peft_model_id} &  [Base ckpt] {args.model_name_or_path}")
        config = PeftConfig.from_pretrained(peft_model_id)
        model = PeftModel.from_pretrained(model, peft_model_id)
    else:
        print(f"Model loaded from {args.model_name_or_path}.", flush=True)

    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()

    passages = src.data.load_passages_xlsx(args.passages)
    
    print(f"Embedding generation for {len(passages)} passages.")

    allids, allembeddings = embed_passages(args, passages, model, tokenizer)

    print(len(passages[0]['summary']))
    entities = [
        allids,  # field id_field
        [pair['metadata']['patent_id'] for pair in passages], # field patent_id
        [pair['title'] for pair in passages], # field title
        [pair['summary'][:10000] for pair in passages], # field text
        allembeddings # field embeddings
    ]

    # create a connection
    create_connection()

    # drop collection if the collection exists
    if has_collection(_COLLECTION_NAME):
        if not args.append_mode:
            drop_collection(_COLLECTION_NAME)
        else:
            collection = Collection(_COLLECTION_NAME)
            get_entity_num(collection)

    if has_collection(_COLLECTION_NAME) and args.append_mode:
        print("Append mode!")
        pass 
    else:    
        # create collection
        collection = create_collection(_COLLECTION_NAME, _ID_FIELD_NAME, _VECTOR_FIELD_NAME)
        # collection = create_collection(_COLLECTION_NAME,_VECTOR_FIELD_NAME)

        # alter ttl properties of collection level
        set_properties(collection)

    # show collections
    list_collections()

    # insert 10000 vectors with 128 dimension
    # vectors = insert(collection, 10000, _DIM)
    vectors = insert(collection, entities)

    collection.flush()
    # get the number of entities
    get_entity_num(collection)

    # create index
    create_index(collection, _VECTOR_FIELD_NAME)

    # load data to memory
    load_collection(collection)

    # search
    # search(collection, _VECTOR_FIELD_NAME, _ID_FIELD_NAME, vectors[:3])
    search(collection, _VECTOR_FIELD_NAME, vectors[:3])

    # release memory
    release_collection(collection)

    # # drop collection index
    # drop_index(collection)

    # # drop collection
    # drop_collection(_COLLECTION_NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--passages", type=str, default=None,
                        help="Path to passages (.tsv file)")
    parser.add_argument("--prefix", type=str, default="passages",
                        help="prefix path to save embeddings")

    parser.add_argument(
        "--per_gpu_batch_size", type=int, default=512, help="Batch size for the passage encoder forward pass"
    )
    parser.add_argument("--passage_maxlength", type=int,
                        default=512, help="Maximum number of tokens in a passage")
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true",
                        help="inference in fp32")
    parser.add_argument("--no_title", action="store_true",
                        help="title not added to the passage body")
    parser.add_argument("--lowercase", action="store_true",
                        help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true",
                        help="lowercase text before encoding")
    parser.add_argument("--append_mode", action="store_true",
                    help="append additional data for existing collection")

    parser.add_argument("--use_peft", action="store_true", help="PEFT mode")
    parser.add_argument("--peft_model_path", type=str, help="PEFT path")
    
    args = parser.parse_args()

    src.slurm.init_distributed_mode(args)

    main(args)
