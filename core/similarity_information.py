## Milvus retrieves similar titles

from pymilvus import MilvusClient
from utils import load_yaml_conf
from sentence_transformers import SentenceTransformer


EMBED_MODEL_NAME = './m3e-large'
EMBED_DEVICE = 'cuda:0'
RECALL_NUM = 20
UPPER_BOUND = 0.95
LOWER_BOUND = 0.50
LIMIT = 5


def search_simi_information(text, embed_model):
    # embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)

    info = load_yaml_conf('../configs/milvus.yaml')
    client = MilvusClient(
        uri=info['uri'],
        user=info['user'],
        password=info['password'],
        db_name=info['database']
    )

    text_embedding = embed_model.encode(text)
    res = client.search(
        collection_name='rumor_info', 
        data=[text_embedding],
        limit=RECALL_NUM,
        search_params={
            "metric_type": "COSINE",
            "params": {}
        },
        output_fields=["title", "content"]
    )

    simi_text_ls = []
    for item in res[0]:
        if UPPER_BOUND > item['distance'] > LOWER_BOUND:
            simi_text_ls.append(
                {
                    "title": item['entity']['title'].strip(),
                    "content": item['entity']['content'].strip()
                }
            )

    return simi_text_ls[:LIMIT]

