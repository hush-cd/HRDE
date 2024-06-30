## Perform batch testing on HRDE using evaluator.py.

import os

from sentence_transformers import SentenceTransformer

from evaluator import Evaluator
from main import RefuteRumor
from es import Es
from milvus import Milvus
from utils import load_yaml_conf
from llm import (
    Qwen_4B_Chat, 
    Qwen_14B_Chat, 
    ChatGLM3_6B, 
    Baichuan2_13B_Chat,
    GPT_transit
)
from data_loader import Dataset


ES_CONF_PATH = '../configs/es.yaml'
MILVUS_CONF_PATH = '../configs/milvus.yaml'
EMBED_MODEL_NAME = './m3e-large'
EMBED_DEVICE = 'cuda:0'
DEV_DATA_PATH = '../data/dev_data/dev_data.json'
OUTPUT_DIR = r'../outputs/evaluation_result'
DATA_NUM = 2500
MODEL1_NAME = "Qwen_14B_Chat_SFT"
MODEL2_NAME = "Qwen_4B_Chat_SFT"


es_info = load_yaml_conf(ES_CONF_PATH)
reference_es = Es(
    host=es_info['host'],
    port=es_info['port'],
    user=es_info['user'],
    password=es_info['password'],
    index=es_info['index']
)

info = load_yaml_conf(MILVUS_CONF_PATH)
reference_milvus = Milvus(
    uri=info['uri'],
    user=info['user'],
    password=info['password'],
    database=info['database'],
    collection=info['collection']
)

model1 = Qwen_14B_Chat(model_name=MODEL1_NAME, 
                      temperature=0.1, 
                      prompt_dir='../prompts/generate')
model2 = Qwen_4B_Chat(model_name=MODEL2_NAME, 
                      temperature=0.1, 
                      prompt_dir='../prompts/generate')
embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)

data = Dataset(path_or_data=DEV_DATA_PATH, data_name=os.path.basename(DEV_DATA_PATH))

evaluator = Evaluator(
    dev_data=data.resize(DATA_NUM),
    model1=model1,
    model2=model2,
    reference_es=reference_es,
    reference_milvus=reference_milvus,
    embedding_model=embed_model,
    embedding_device=EMBED_DEVICE,
    k=5,
    k_keyword_recall=5,
    k_vector_recall=25,
    keywords_num=20,
    similarity_threshold=0.5,
    with_reference=True,
    chunk_size=250,
    process_num=10,
    process_num_model1=10,
    output_dir=OUTPUT_DIR
)

print('start')
results = evaluator.run()