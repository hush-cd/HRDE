## Import reference data into the Elastic search.

import os

import pandas as pd
from es import Es
from utils import load_yaml_conf
from reference_data_process import ReferenceDataProcess
from sentence_transformers import SentenceTransformer


EMBED_MODEL_NAME = './m3e-large'  # embedding model
EMBED_DEVICE = 'cuda:0'
REF_DATA_PATH = '../data/reference_data'  # file path of original reference data
ES_CONF_PATH = '../configs/es.yaml'


embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)

es_info = load_yaml_conf(ES_CONF_PATH)
reference_es = Es(
    host=es_info['host'],
    port=es_info['port'],
    user=es_info['user'],
    password=es_info['password'],
    index=es_info['index']
)

data_files = os.listdir(REF_DATA_PATH)
cnt = 0
for df in data_files:
    print(f'{df} start ...')
    rdp = ReferenceDataProcess(
        path=os.path.join(REF_DATA_PATH, df),
        embedding_model=embed_model
    )
    num = rdp.process_for_es()
    rdp.add_to_es(es=reference_es)
    cnt += num
    print(f'{df} finished！')
    print(cnt)