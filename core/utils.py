import logging
import os
from typing import List, Union

import jieba
import jieba.analyse
import numpy as np
import yaml
from loguru import logger


jieba.setLogLevel(logging.INFO)
STOPWORDS_FILE = '../stopwords'



def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """向量余弦相似度"""
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    return dot_product / (norm_u * norm_v)


def get_stopwords() -> set:
    """获取停用词表"""
    files = os.listdir(STOPWORDS_FILE)
    for f in files:
        stopwords = []
        with open(os.path.join(STOPWORDS_FILE, f), 'r') as fp:
            stopwords.extend(fp.read().split('\n'))
    stopwords = set(stopwords)

    return stopwords


def keyword_extraction(sentence: str, k: int = 20, stopwords: Union[set, list] = None):
    """关键词提取"""
    keywords1 = jieba.analyse.extract_tags(sentence, topK=k)
    keywords2 = jieba.analyse.textrank(sentence, topK=k)
    keywords_set = set(keywords1 + keywords2)
    keywords1_record = {w: i for i, w in enumerate(keywords1)}
    keywords2_record = {w: i for i, w in enumerate(keywords2)}

    keywords = []
    for w in keywords_set:
        if stopwords and w in stopwords:
            continue
        score = k * 2 - (keywords1_record.get(w, k) + keywords2_record.get(w, k))
        keywords.append((w, score))
    
    keywords.sort(key=lambda x: x[1], reverse=True)

    return [item[0] for item in keywords[:k]]


def load_yaml_conf(yaml_path: str):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        return config_data
    except FileNotFoundError:
        logger.error(f"File '{yaml_path}' not found.")
    except yaml.YAMLError as e:
        logger.error(f"Unable to load YAML file '{yaml_path}': {e}")