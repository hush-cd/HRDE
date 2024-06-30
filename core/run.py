# main调用示例
from main import RefuteRumor
from es import Es
from milvus import Milvus
from utils import load_yaml_conf
from llm import (
    Qwen_4B_Chat, 
    Qwen_14B_Chat
)
from sentence_transformers import SentenceTransformer

ES_CONF_PATH = '../configs/es.yaml'
MILVUS_CONF_PATH = '../configs/milvus.yaml'
EMBED_MODEL_NAME = './m3e-large'
EMBED_DEVICE = 'cuda:0'
SIMILARITY_THRESHOLD = 0.4


def query(input):
    # 连接 es 引擎
    es_info = load_yaml_conf(ES_CONF_PATH)
    reference_es = Es(
        host=es_info['host'],
        port=es_info['port'],
        user=es_info['user'],
        password=es_info['password'],
        index=es_info['index']
    )

    # 连接 milvus 向量数据库
    info = load_yaml_conf(MILVUS_CONF_PATH)
    reference_milvus = Milvus(
        uri=info['uri'],
        user=info['user'],
        password=info['password'],
        database=info['database'],
        collection=info['collection']
    )

    # 初始化大模型和嵌入模型
    model1 = Qwen_14B_Chat(temperature=0.1, prompt_dir='../prompts/generate')
    model2 = Qwen_4B_Chat(temperature=0.1, prompt_dir='../prompts/generate')
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)
    # 开始引证回答
    rr = RefuteRumor(
        model1=model1,
        model2=model2,
        reference_es=reference_es,
        reference_milvus=reference_milvus,
        embedding_model=embed_model,
        embedding_device=EMBED_DEVICE
    )

    r = rr.run(input, similarity_threshold=SIMILARITY_THRESHOLD)
    
    return r