# Smart Patent Search 

This repository shows how to build a patent search system and create a vector database (MilVus) to run demo using gradio.

## Step1. Set environment
```
$ git clone https://github.com/hanseokOh/PatentSearch.git

# Option1. Build with conda enviroment
$ cd PatentSearch
$ conda create -n patent_search python=3.8
$ conda activate patent_search
$ pip install -r requirements.txt

# Option2. Build with docker container 
$ docker pull hanseokoh/patent_search:latest
$ docker run -v ./PatentSearch:/workspace/directory -it --name patent_search --gpus all hanseokoh/patent_search:latest /bin/bash
```

## Step2. Training (optional if you need)
```
$ cd model
$ bash ./training.sh
```

## Step3. Evaluation 
```
# select checkpoint based on train & validation log
$ bash ./eval.sh
```

## Step4. Indexing (using Milvus Vector Database)
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
$ bash ./indexing.sh
```

## Step5. Run Demo
```
$ python gradio_demo-milvus.py

# When you want to create temporal URL
$ cloudflared tunnel --url http://localhost:7860
```

