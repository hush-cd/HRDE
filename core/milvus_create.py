## Milvus reference data index construction

from pymilvus import (
    MilvusClient, 
    DataType, 
    db, 
    connections, 
    FieldSchema, 
    CollectionSchema
)


# 创建数据库
conn = connections.connect(
    host="",
    user="",
    password="",
    port=19530
)

database = db.create_database("hrde")
db.using_database("hrde")

# 连接milvus
client = MilvusClient(
    uri="",
    user="",
    password="",
    db_name="hrde"
)

# 创建 schema
id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, description="primary id")
title_field = FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1000, description="title")
text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000, description="content chunk")
text_vector_field = FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=1024, description="content chunk embedding")
date_field = FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=10, description="date")
url_field = FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1000, description="url")
source_field = FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1000, description="source")

schema = CollectionSchema(
    fields=[id_field, title_field, text_field, text_vector_field,
            date_field, url_field, source_field], 
    auto_id=True, 
    enable_dynamic_field=True, 
    description="desc of a collection"
)

# 创建索引
index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="text_vector",
    metric_type="COSINE",
    index_type="IVF_FLAT",
    index_name="text_vector_index",
    params={ "nlist": 128 }
)

# 创建集合
client.create_collection(
    collection_name="reference",
    schema=schema,
    index_params=index_params
)