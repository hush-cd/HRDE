## Main function of HRDE.

import hashlib
import math
import re
import os
import copy
from abc import ABC
from typing import Union

from sentence_transformers import SentenceTransformer

from es import Es
from milvus import Milvus
from llm import BaseLLM
from utils import cosine_similarity, get_stopwords, keyword_extraction


class RefuteRumor(ABC):
    """健康信息辟谣"""
    def __init__(
            self,
            model1: BaseLLM,
            model2: BaseLLM,
            reference_es: Es,
            reference_milvus: Milvus,
            embedding_model: SentenceTransformer,
            embedding_device: str = None
    ):
        self.model1 = model1
        self.model2 = model2
        self.reference_es = reference_es
        self.reference_milvus = reference_milvus
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device

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

    @staticmethod
    def _similarity_search(query_embeddings, corpus_embeddings, k, threshold):
        """
        计算两个向量集合的相似性，并排序给出 Top-K ，返回 Top-K 的序号
        query_embeddings: 查询向量
        corpus_embeddings: 待比对向量
        k: 召回数
        threshold: 相似度阈值，低于阈值的待比对向量被剔除
        """
        if not query_embeddings:
            return []
        
        simi_ls = []
        for i, ce in enumerate(corpus_embeddings):
            simi_ = [cosine_similarity(qe, ce) for qe in query_embeddings]
            if max(simi_) < threshold:
                continue
            simi_ls.append((i, sum(simi_)))
        simi_ls.sort(key=lambda x: x[1], reverse=True)
        return [i[0] for i in simi_ls[:k]]
 
    def _split_and_embedding(
            self, 
            text: str, 
            size: int = 500
    ):
        if not text:
            return [self.embedding_model.encode(text)]
        
        n = len(text)
        embeds = []
        for i in range(math.ceil(n/size)):
            # embed = get_embedding(text[i * size:(i+1) * size])
            embed = self.embedding_model.encode(
                sentences=text[i * size:(i+1) * size],
                show_progress_bar=False,
                device=self.embedding_device
            )
            embeds.append(embed)

        return embeds
    
    @staticmethod
    def _get_md5_8(a):
        """获取中文文本 md5 摘要，并调整长度为 8"""
        md5_hash = hashlib.md5(a.encode('utf-8')).digest()
        s = '0123456789ABCDEFGHIJKLMNOPQRSTUV'
        d = ''
        for f in range(8):
            g = md5_hash[f]
            d += s[(g ^ md5_hash[f + 8]) - g & 0x1F]
        return d
    
    def _post_process(self, text, refs):
        # 搜索和归并同一个来源-文章的文档
        cnt = 0
        record = dict()
        for d in refs:
            if d['md5'] not in text:
                continue
            k = (d['source'], d['title'], d['url'])
            if k in record:
                record[k][1].append(d['md5'])
            else:
                cnt += 1
                record[k] = [f'[{cnt}]', [d['md5']]]
        if cnt == 0:
            return text

        # 记录参考文献字符串，并反转 record
        ref_str = []
        record_reverse = dict()
        for k in record:
            sub_code, code_ls = record[k]
            for c in code_ls:
                record_reverse[c] = sub_code
            ref_str.append(f"{sub_code}《{k[1]}》- 来源：{k[0]}（ {k[2]} ）\n")

        # 匹配出现的引用的字符串
        pattern = re.compile(r"(?:\[[0-9A-Z]{8}\])+")
        matches = pattern.finditer(text)

        # 重新整理连续多个引用编码的排列（去重、排序）
        record2 = []
        for match in matches:
            l, r = match.span()
            # index_str = match.group()
            # index_ls = [i + ']' for i in match.group().split(']') if i]
            # index_ls = sorted(set([record_reverse[i] for i in index_ls]))
            # record2.append((l, r, ''.join(index_ls)))
            index_ls = [i + ']' for i in match.group().split(']') if i]
            new_index_ls = []
            for i in index_ls:
                if i in record_reverse:
                    new_index_ls.append(record_reverse[i])
            new_index_ls = sorted(set(new_index_ls))
            record2.append((l, r, ''.join(new_index_ls)))

        # 替换
        for item in record2[::-1]:
            l, r, index_str = item
            text = text[:l] + index_str + text[r:]
        
        text += "\n\n[参考资料]\n" + ''.join(ref_str)
        return text
    
    def run(
            self, 
            input: str,
            k: int = 5,
            k_keyword_recall: int = 5,
            k_vector_recall: int = 25,
            keywords_num: int = 20,
            similarity_threshold: float = 0.5,
            with_reference: bool = True,
            chunk_size: int = 250,
            stream: bool = False
    ):
    """
    main method of refute a rumor.

    Parameters:
    -----------
    input : str
        The input health information text.
    k : int, optional
        The maximum number of reference document chunks to accept. Default is 5.
    k_keyword_recall : int, optional
        The number of top keyword-based recalls to consider. Default is 5.
    k_vector_recall : int, optional
        The number of top vector-based recalls to consider. Default is 25.
    keywords_num : int, optional
        The number of keywords to extract from the input text. Default is 20.
    similarity_threshold : float, optional
        The threshold for similarity score to filter reference document chunks. Default is 0.5.
    with_reference : bool, optional
        Whether to provide reference documents. Default is True.
    chunk_size : int, optional
        The size of chunks to split the reference document from Elasticsearch. Default is 250.
    stream : bool, optional
        Whether to use streaming generation for large model output. Default is False.

    Returns:
    --------
    dict: A dictionary containing the type of response, prompt used, and the answer, with or without references.
        - If "with_reference" is True and there are reference documents:
            {
                "type": "with_reference",
                "answer": answer with reference documents,
                "ref_items": reference documents
            }
            or
            {
                "type": "with_reference",
                "prompt": prompt,
                "origin_answer": original answer reference documents,
                "answer": answer reference documents
            }
        - If "with_reference" is True and there are no reference documents:
            {
                "type": "without_reference",
                "answer": answer without reference documents
            }
            or
            {
                "type": "without_reference",
                "prompt": prompt,
                "answer": answer without reference documents
            }
    
    """
        input_len = len(input)
        if input_len > 1500:
            raise ValueError('The input length must be less than 1500!')
        
        # 无引证回答
        prompt, answer = self.model2.refute_rumor((input, ''))

        # 如果不采用引证资料，则直接输出无引证回答
        if not with_reference:
            return {
                "type": "without_reference",
                "prompt": prompt,
                "answer": answer
            }

        # 输入、无引证回答向量化
        input_embeddings = self._split_and_embedding(input)
        answer_embeddings = self._split_and_embedding(answer)
        # 输入关键词抽取
        stopwords = get_stopwords()
        keywords = keyword_extraction(input, k=keywords_num, stopwords=stopwords)

        # 基于关键词的文档召回
        docs1 = self.reference_es.search(
            {
                "method": 'match',
                "field": 'text',
                "q": ' '.join(keywords),
                "size": k_keyword_recall
            }
        )
        # 对文章进行分块
        new_items = []
        for item in docs1['items']:
            sentences = self._text_split(item['text'])
            chunks = self._gene_chunk(sentences, chunk_size=chunk_size)
            for chunk in chunks:
                new_item = copy.deepcopy(item)
                new_item['text'] = chunk
                new_items.append(new_item)
        docs1['items'] = new_items
        docs1['num'] = len(new_items)
        
        # 基于向量的文档召回
        docs2 = self.reference_milvus.search(
            {
                "query_vector": input_embeddings,
                "limit": k_vector_recall
            }
        )
        
        # 整合两路召回的文档，基于与无引证回答的相似度进行排序和筛选
        docs = []
        # 去重
        for doc in docs1['items'] + docs2['items']:
            if doc not in docs:
                docs.append(doc)
        docs_embeddings = self.embedding_model.encode(
            sentences=[doc['text'] for doc in docs],
            batch_size=32,
            show_progress_bar=False,
            device=self.embedding_device
        )
        index_ls = self._similarity_search(
            answer_embeddings,
            docs_embeddings,
            k=k,
            threshold=similarity_threshold
        )
        
        docs_remain = [docs[i] for i in index_ls]
        
        if docs_remain:
            docs_str = []
            for d in docs_remain:
                d_md5 = self._get_md5_8(d['text'])
                d['md5'] = f'[{d_md5}]'
                docs_str.append(f"[{d_md5}]《{d['title']}》：{d['text']}")

            if stream:
                prompt = self.model1.process_prompt((input, docs_str))
                answer_with_refs = self.model1.stream_request(prompt)
                return {
                    "type": "with_reference",
                    "answer": answer_with_refs,
                    "ref_items": docs_remain
                }

            prompt, answer_with_refs = self.model1.refute_rumor((input, docs_str))
            answer_with_refs2 = self._post_process(answer_with_refs, docs_remain)
            return {
                "type": "with_reference",
                "prompt": prompt,
                "origin_answer": answer_with_refs,
                "answer": answer_with_refs2
            }
        else:
            if stream:
                prompt = self.model1.process_prompt((input, ''))
                answer_without_refs = self.model1.stream_request(prompt)
                return {
                    "type": "without_reference",
                    "answer": answer_without_refs
                }
                
            prompt, answer = self.model1.refute_rumor((input, ''))
            return {
                "type": "without_reference",
                "prompt": prompt,
                "answer": answer
            }
        

if __name__ == '__main__':
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


    es_info = load_yaml_conf(ES_CONF_PATH)
    reference_es = Es(
        host=es_info['host'],
        port=es_info['port'],
        user=es_info['user'],
        password=es_info['password'],
        index=es_info['index']
    )

    info = load_yaml_conf(MILVUS_CONF_PATH)
    reference_milvus = Milvus(
        uri=info['uri'],
        user=info['user'],
        password=info['password'],
        database=info['database'],
        collection=info['collection']
    )

    model1 = Qwen_14B_Chat(temperature=0.1, prompt_dir='../prompts/generate')
    model2 = Qwen_4B_Chat(temperature=0.1, prompt_dir='../prompts/generate')
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)

    rr = RefuteRumor(
        model1=model1,
        model2=model2,
        reference_es=reference_es,
        reference_milvus=reference_milvus,
        embedding_model=embed_model,
        embedding_device=EMBED_DEVICE
    )

    input = '珍珠奶茶中的珍珠有剧毒'
    r = rr.run(input, similarity_threshold=SIMILARITY_THRESHOLD)