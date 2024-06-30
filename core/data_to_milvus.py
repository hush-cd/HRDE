## Import reference data into the Milvus

import os
import pandas as pd
import milvus
from reference_data_process import ReferenceDataProcess
from utils import load_yaml_conf
from sentence_transformers import SentenceTransformer


EMBED_MODEL_NAME = './m3e-large'  # embedding model
EMBED_DEVICE = 'cuda:0'
REF_DATA_PATH = '../data/reference_data'  # # file path of original reference data
MILVUS_CONF_PATH = '../configs/milvus.yaml'


embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)

info = load_yaml_conf(MILVUS_CONF_PATH)
client = milvus.Milvus(
    uri=info['uri'],
    user=info['user'],
    password=info['password'],
    database=info['database'],
    collection=info['collection']
)

data_files = os.listdir(REF_DATA_PATH)
cnt = 0
for df in data_files:
    print(f'{df} start ...')
    rdp = ReferenceDataProcess(
        path=os.path.join(REF_DATA_PATH, df),
        embedding_model=embed_model
    )
    num = rdp.process_for_milvus(show_progress_bar=True, device=EMBED_DEVICE)
    rdp.add_to_milvus(client)
    cnt += num
    print(f'{df} finished!')
    print(cnt)