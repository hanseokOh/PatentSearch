# Smart Patent Search 

This repository shows how to build a patent search system and create a vector database (MilVus) to run demo using gradio.

You can also download our model in huggingface hub [link](https://huggingface.co/hanseokOh/smartPatent-mContriever-lora). 

If you just want to use our demo without training from scratch, go to [Step1.5](#step1.5) and then [Step5](#step5). We already provide related file in repo.

## Step1. Set environment
```
$ git clone https://github.com/hanseokOh/PatentSearch.git

# Option1. Build with conda enviroment
$ cd PatentSearch
$ conda create -n patent_search python=3.8
$ conda activate patent_search
$ pip install -r requirements.txt

# Option2. Build with docker container [TBD - under fix]
(base image : pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel)
$ docker pull hanseokoh/patent_search:latest
$ docker run -v ./PatentSearch:/workspace/directory -it --name patent_search --gpus all hanseokoh/patent_search:latest /bin/bash
```
<a name="step1.5"></a>
## Step1.5. Download dataset 
```

# Option1. Manual downloading
https://drive.google.com/drive/folders/1ExqLPJ5O0sM490DobNBOrkleMhC602XD?usp=sharing

# Option2. download with gdown python library
$ pip install gdown
$ gdown --folder 1ExqLPJ5O0sM490DobNBOrkleMhC602XD

Then move downloaded files under data directory. 

```

## Step2. Training (optional if you need)
```
$ cd model
# for full finetuning
$ bash ./run.sh

# for PEFT tuning - we use LR:5e-4 as final model
$ bash ./peft_run.sh
```

## Step3. Evaluation 
```
# select checkpoint based on train & validation log
$ bash ./eval.sh
```

## Step4.A Indexing (using Milvus Vector Database)
```
# download Milvus
$ wget https://github.com/milvus-io/milvus/releases/download/v2.3.1/milvus-standalone-docker-compose.yml -O docker-compose.yml

# Run Milvus (daemon)
$ docker compose up -d

# Check Milvus container is on
$ docker compose ps 

# Access test 
$ docker port milvus-standalone 19530/tcp

# Down server (After use)
$ docker compose down


-----------------
# Create Vector DB using trained retriever
$ cd model
$ bash ./indexing.sh
```
## (Optional) Step4.B Indexing (w/o Milvus - locally saving)
```
$ cd model
$ bash ./local_indexing.sh
```

<a name="step5"></a>
## Step5. Run Demo
```
###  Using Milvus
- Search
$ python gradio_search_milvus.py

- Search & Generate (RAG)
$ python gradio_rag_milvus.py

###  w/o Milvus
- Search
$ python gradio_search.py

- Search & Generate (RAG)
$ python gradio_rag_local.py


# (Optional) When you want to create temporal URL (need to install cloudflared package)
$ cloudflared tunnel --url http://localhost:7860
```

