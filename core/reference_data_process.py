import os
import sys
import re
import math
from abc import ABC

import pandas as pd
from sentence_transformers import SentenceTransformer

from milvus import Milvus
from es import Es
# from utils import get_embedding


class ReferenceDataProcess(ABC):
    """
    引证数据处理以及导入 ES 引擎
    path: 爬取的数据文件(.csv)，字段包括['title', 'text', 'date', 'url']
    source: 数据来源，如果为 None ，则使用 path 中的数据文件名作为 source
    """
    def __init__(
            self, 
            path: str,
            embedding_model: SentenceTransformer,
            source: str = None
    ):
        name, tail = os.path.splitext(os.path.split(path)[-1])
        if tail != '.csv':
            raise ValueError('path must be a .csv file.')
        
        data = pd.read_csv(path)
        if set(data.columns) != {'title', 'text', 'date', 'url'}:
            raise KeyError("Missing field, field must be ['title', 'text', 'date', 'url'].")
        self.data = data[['title', 'text', 'date', 'url']]
        
        self.source = source if source else name
        self.embedding_model = embedding_model
    
    def _is_contain_chinese(self, string):
        """判断字符串是否不包含中文"""
        if len(re.findall('[^\u4e00-\u9fa5]', string)) == len(string):
            return False
        else:
            return True

    def _text_split(self, text, size=80):
        """中文文本分句，单句超过 80 的会被切分"""
        patterns = [r"""([。！？\?\!])([^”'"])""",
                    r"""(\.{6})([^”'"])""",
                    r"""(\…{2})([^”'"])""",
                    r"""([\.{6}\…{2}。！？\?\!][”'"])([^，。！？\?\!])"""]
        for p in patterns:
            text = re.sub(p, r"\1\n\2", text)
        
        sentences = []
        for s in text.split('\n'):
            if s and self._is_contain_chinese(s):
                length = len(s)
                for i in range(math.ceil(length/size)):
                    sentences.append(s[i * size:(i+1) * size])
        return sentences

    def _gene_chunk(self, sentences, chunk_size=250):
        """基于句子列表生成文本块，可控制块的大小"""
        cur_chunk = ''
        cur_size = 0
        chunks = []
        for s in sentences:
            cur_size += len(s)
            cur_chunk += s
            if cur_size >= chunk_size * 0.95:
                chunks.append(cur_chunk)
                cur_chunk = ''
                cur_size = 0
        if cur_chunk:
            chunks.append(cur_chunk)
        return chunks

    def process_for_milvus(
            self, 
            chunk_size: int = 250, 
            batch_size: int = 32,
            show_progress_bar: bool = None,
            device: str = None

    ):
        """处理数据（字段处理、分块、向量化）以形成可以导入 milvus 的格式"""
        # 剔除有缺失值的行，并且将索引重新整理
        self.data.dropna(axis=0, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

        docs = []
        all_chunk = []
        for i, row in self.data.iterrows():
            title, text, date, url = row
            title = title.strip()
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)
            date = date.strip() if type(date) == str else date
            url = url.strip()
            sentences = self._text_split(text)
            chunks = self._gene_chunk(sentences, chunk_size=chunk_size)
            for chunk in chunks:
                document={
                            "title": title,
                            "text": chunk,
                            "url": url,
                            "source": self.source,
                            "date": date,
                        }
                docs.append(document)
            
            all_chunk.extend(chunks)
        
        chunk_embeddings = self.embedding_model.encode(
            sentences=all_chunk,
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar,
            device=device
        )
        
        for i in range(len(chunk_embeddings)):
            docs[i]['text_vector'] = chunk_embeddings[i]

        self.docs_for_milvus = docs

        return len(docs)
    
    def process_for_es(self):
        # 剔除有缺失值的行，并且将索引重新整理
        self.data.dropna(axis=0, inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        docs = []
        for i, row in self.data.iterrows():
            title, text, date, url = row
            title = title.strip()
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)
            date = date.strip() if type(date) == str else date
            url = url.strip()
            document={
                        "text": "[title]" + title + "[title]" + "[text]" + text + "[text]",
                        "url": url,
                        "source": self.source,
                        "date": date,
                    }
            docs.append(document)

        self.docs_for_es = docs

        return len(docs)
        

    def add_to_milvus(self, client: Milvus):
        """将数据导入 milvus """
        return client.add_data(data=self.docs_for_milvus)
    
    def add_to_es(self, es: Es):
        """将数据导入 ES 引擎指定 index"""
        return es.add_many_data(documents=self.docs_for_es)
        