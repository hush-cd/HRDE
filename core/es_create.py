## Elasticsearch reference data index construction

from es import Es

es = Es(
    host='',
    port=9200,
    user="",
    password=""
)

body = {
    "settings" : {
        "index" : {
            "number_of_shards" : 1,
            "number_of_replicas" : 1
        }
    },
    "mappings": {
        "properties": {
            "text": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
                },
            "url": {
                "type": "keyword"
                },
            "source": {
                "type": "keyword"
                },
            "date": {
                "type": "date"
                },
        }
    }
}
es.create_index(
    index='reference', 
    settings=body["settings"], 
    mappings=body["mappings"]
)