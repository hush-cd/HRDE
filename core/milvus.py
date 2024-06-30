import math
from typing import List
from pymilvus import MilvusClient


BATCH_SIZE = 1000


class Milvus:
    def __init__(self, uri, user, password, database, collection):
        self.database = database
        self.collection = collection
        self.client = MilvusClient(
            uri=uri,
            user=user,
            password=password,
            db_name=database
        )

    def get_collections(self):
        return self.client.list_collections()
    
    def search(self, query: dict):
        query_vector = query['query_vector']
        limit = query.get('limit', 20)
        metric_type = query.get('metric_type', "COSINE")
        vector_num = len(query_vector)
        batch_size = math.ceil(limit / vector_num)
        res_all = self.client.search(
            collection_name=self.collection, 
            data=query_vector,
            limit=batch_size,
            search_params={
                "metric_type": metric_type,
                "params": {}
            },
            output_fields=["title", "text", "date", "url", "source"]
        )

        items = []
        for res in res_all:
            for item in res:
                new_item = item['entity']
                items.append(new_item)
        
        items = items[:limit]
        res_format = {
            'num': len(items),
            'items': items
            }

        return res_format
    
    def add_data(self, data: List[dict]):
        num = len(data)
        batch_num = math.ceil(num / BATCH_SIZE)

        res = {'insert_count': 0}
        for i in range(batch_num):
            batch_res = self.client.insert(
                collection_name=self.collection,
                data=data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            )
            res['insert_count'] += batch_res['insert_count']
        return res