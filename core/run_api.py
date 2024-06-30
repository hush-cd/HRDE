# main调用示例
from main import RefuteRumor
from es import Es
from milvus import Milvus
from utils import load_yaml_conf
from llm import Qwen_14B_Chat, Qwen_4B_Chat
from sentence_transformers import SentenceTransformer
import json
from flask_cors import CORS
import random
from similarity_information import search_simi_information
import logging
import functools
from logging.handlers import TimedRotatingFileHandler

from flask import Flask, jsonify, request, g, Response, stream_with_context

app = Flask(__name__)

cors = CORS(app, resources={r"*": {"origins": "*"}})

ES_CONF_PATH = '../configs/es.yaml'
MILVUS_CONF_PATH = '../configs/milvus.yaml'
EMBED_MODEL_NAME = './m3e-large'
EMBED_DEVICE = 'cuda:0'

query_list = []
with open('./hot.txt') as file:
    for line in file:
        query_list.append(line.strip())

embedding_model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)


logger = logging.getLogger('request_logger')
logger.setLevel(logging.INFO)
log_file = '/data/log/app.log'  # 指定日志文件路径
file_handler = TimedRotatingFileHandler(log_file, when='midnight', interval=1, backupCount=0)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def log_request(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 记录来源 IP
        if request.headers.getlist("X-Forwarded-For"):
            source_ip = request.headers.getlist("X-Forwarded-For")[0]
        else:
            source_ip = request.remote_addr
        # source_ip = request.remote_addr
        # 记录输入
        if request.method == 'POST':
            input_data = request.get_json() if request.is_json else request.form.to_dict()
        else:
            input_data = request.args.to_dict()

        # 调用原始路由函数
        response = func(*args, **kwargs)

        # 记录输出
        output_data = response.get_json() if response is not dict else response

        # 打印日志（可以改为记录到文件或数据库）
        logger.info(f"Source IP: {source_ip}, Input: {input_data}, Output: {output_data}")

        return response
    return wrapper

@app.route('/query_recommend', methods=['GET'])
@log_request
def query_recommend():
    random.shuffle(query_list)
    response = jsonify({"data": query_list[:5]})
    return response

@app.route('/inputs_recommend', methods=['POST'])
@log_request
def inputs_recommend():
    inputs = request.get_json().get('input')
    data = search_simi_information(inputs, embedding_model)
    response = jsonify({"data": data})
    return response

@app.route('/query', methods=['POST'])
@log_request
def query():
    input = request.get_json().get('input')
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
    model2 = Qwen_4B_Chat(temperature=0.1, prompt_dir='../prompts/generate')
    model1 = Qwen_14B_Chat(temperature=0.1, prompt_dir='../prompts/generate')

    # 开始引证回答
    rr = RefuteRumor(
        model1=model1,
        model2=model2,
        reference_es=reference_es,
        reference_milvus=reference_milvus,
        embedding_model=embedding_model,
        embedding_device=EMBED_DEVICE
    )

    r = rr.run(input, similarity_threshold=0.4, stream=True)


    def process_res(res):
        if res.get('type') != 'with_reference':
            # data = {'type': 'answer', "data": res.get('answer')}
            for item in res.get('answer'):
                if item:
                    data = {'type': 'answer', "data": item[0]}
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            # yield f"data: {json.dumps(data, ensure_ascii=False) }\n\n"
            yield f"data: DONE\n\n"
        else:
            data = {'type': 'reference', "data": res.get('ref_items')}
            yield f"data: {json.dumps(data, ensure_ascii=False) }\n\n"
            
            for item in res.get('answer'):
                if item:
                    data = {'type': 'answer', "data": item[0]}
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            yield f"data: DONE\n\n"

    headers = {"Content-Type": "text/event-stream"}
    resp = Response(stream_with_context(process_res(r)), headers=headers)

    return resp


if "__main__" == __name__:
    app.run(host="0.0.0.0", port=10002)
