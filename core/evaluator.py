## Conduct batch testing on HRDE and record the responses on each test sample.

import os
import re
import json
import math
import copy
import datetime
import hashlib
from abc import ABC
from typing import Dict, List, Tuple
from multiprocessing import Pool
from functools import partial

from loguru import logger
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from llm import BaseLLM
from es import Es
from milvus import Milvus
from data_loader import Dataset
from utils import get_stopwords, keyword_extraction, cosine_similarity


os.environ["TOKENIZERS_PARALLELISM"] = "true"
OUTPUT_DIR = r'../outputs/evaluation_result'


class Evaluator(ABC):
    """评估"""
    def __init__(
            self, 
            dev_data: Dataset, 
            model1: BaseLLM,  # 大模型
            model2: BaseLLM,  # 小模型
            reference_es: Es,
            reference_milvus: Milvus,
            embedding_model: SentenceTransformer,
            embedding_device: str = None,
            k: int = 5,
            k_keyword_recall: int = 5,
            k_vector_recall: int = 25,
            keywords_num: int = 20,
            similarity_threshold: float = 0.7,
            with_reference: bool = True,
            chunk_size: int = 250,
            process_num: int = 1,
            process_num_model1: int = 1,
            output_dir: str = OUTPUT_DIR
    ):
        # 评估数据集
        self.data = dev_data.load()
        self.data_name = dev_data.data_name

        # 评估所使用的大模型、嵌入模型、搜索引擎和向量数据库
        self.model1 = model1
        self.model2 = model2
        self.embedding_model = embedding_model
        self.embedding_device = embedding_device
        self.reference_es = reference_es
        self.reference_milvus = reference_milvus

        # 评估所使用的参数
        self.k = k
        self.k_keyword_recall = k_keyword_recall
        self.k_vector_recall = k_vector_recall
        self.keywords_num = keywords_num
        self.similarity_threshold = similarity_threshold
        self.with_reference = with_reference
        self.chunk_size = chunk_size

        # 其他运行和文件保存配置
        self.stopwords = get_stopwords()
        self.process_num = process_num
        self.process_num_model1 = process_num_model1
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model1_name = self.model1.params['model_name']
        model2_name = self.model2.params['model_name']
        output_name = f'{model1_name}_{model2_name}_{{with_refs}}_{timestamp}.json'
        self.output_path = os.path.join(output_dir, output_name)

    def evaluate_info(self):
        return {
            "dev_data": self.data_name,
            "model1": self.model1.params['model_name'],
            "model2": self.model2.params['model_name'],
            "elastic_index": self.reference_es.index,
            "milvus_collection": f'{self.reference_milvus.database}:{self.reference_milvus.collection}',
            "k": self.k,
            "k_keyword_recall": self.k_keyword_recall,
            "k_vector_recall": self.k_vector_recall,
            "keywords_num": self.keywords_num,
            "similarity_threshold": self.similarity_threshold,
            "chunk_size": self.chunk_size,
            "with_reference": self.with_reference,
            "process_num": self.process_num,
            "datetime": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "output_path": self.output_path
        }
    
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
        # flag = False
        for match in matches:
            l, r = match.span()
            # index_str = match.group()
            index_ls = [i + ']' for i in match.group().split(']') if i]
            # index_ls = sorted(set([record_reverse[i] for i in index_ls]))
            new_index_ls = []
            for i in index_ls:
                if i in record_reverse:
                    new_index_ls.append(record_reverse[i])
                # else:
                #     flag = True
            new_index_ls = sorted(set(new_index_ls))
            record2.append((l, r, ''.join(new_index_ls)))

        # 替换
        for item in record2[::-1]:
            l, r, index_str = item
            text = text[:l] + index_str + text[r:]
        
        text += "\n\n[参考资料]\n" + ''.join(ref_str)
        return text
    
    def batch_query_llm1(
            self, 
            inputs: list,
            references: list = None,
            desc: str = ''
    ):
        num = len(inputs)
        if references:
            input_infos = zip(inputs, references)
        else:
            input_infos = zip(inputs, [None] * num)
        with Pool(self.process_num_model1) as pool:
            results = list(tqdm(
                pool.imap(self.model1.refute_rumor, input_infos),
                total = len(inputs),
                desc = self.model1.params['model_name'] + f'({desc})'
            ))

        prompts = [r[0] for r in results]
        answers = [r[1] for r in results]
        return prompts, answers
    
    def batch_query_llm2(
            self, 
            inputs: list,
            references: list = None,
            desc: str = ''
    ):
        num = len(inputs)
        if references:
            input_infos = zip(inputs, references)
        else:
            input_infos = zip(inputs, [None] * num)
        with Pool(self.process_num) as pool:
            results = list(tqdm(
                pool.imap(self.model2.refute_rumor, input_infos),
                total = len(inputs),
                desc = self.model2.params['model_name'] + f'({desc})'
            ))

        prompts = [r[0] for r in results]
        answers = [r[1] for r in results]
        return prompts, answers
    
    def run(self):

        # 生成无引证回答
        inputs = [item['input'] for item in self.data]
        prompts, answers = self.batch_query_llm2(inputs=inputs, desc=f'answer w/o references(model2)')

        num = len(inputs)

        has_reference_index = []
        docs_remain_ls = []
        docs_str_ls = []
        input_embeddings_ls = []
        answer_embeddings_ls = []
        for i in tqdm(range(num), 'docs retrieval and processing'):
            input = inputs[i]
            answer = answers[i]

            # 输入、无引证回答向量化
            input_embeddings = self._split_and_embedding(input)
            answer_embeddings = self._split_and_embedding(answer)

            input_embeddings_ls.append(input_embeddings)
            answer_embeddings_ls.append(answer_embeddings)
            # 输入关键词抽取
            keywords = keyword_extraction(
                input, 
                k=self.keywords_num, 
                stopwords=self.stopwords
            )

            # 基于关键词的文档召回
            docs1 = self.reference_es.search(
                {
                    "method": 'match',
                    "field": 'text',
                    "q": ' '.join(keywords),
                    "size": self.k_keyword_recall
                }
            )
            # 对文章进行分块
            new_items = []
            for item in docs1['items']:
                sentences = self._text_split(item['text'])
                chunks = self._gene_chunk(
                    sentences,
                    chunk_size=self.chunk_size
                )
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
                    "limit": self.k_vector_recall
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
                k=self.k,
                threshold=self.similarity_threshold
            )
            
            docs_remain = [docs[ii] for ii in index_ls]
            if docs_remain:
                docs_str = []
                for d in docs_remain:
                    d_md5 = self._get_md5_8(d['text'])
                    d['md5'] = f'[{d_md5}]'
                    docs_str.append(f"[{d_md5}]《{d['title']}》：{d['text']}")
                has_reference_index.append(i)
                docs_remain_ls.append(docs_remain)
                docs_str_ls.append(docs_str)
        
        # 生成有引证回答
        inputs2 = [inputs[i] for i in has_reference_index]
        prompts_with_refs, answers_with_refs = self.batch_query_llm1(
            inputs=inputs2,
            references=docs_str_ls, 
            desc='answer with references(model1)'
        )

        answers_with_refs2 = []
        for i in range(len(inputs2)):
            answers_with_refs2.append(self._post_process(answers_with_refs[i], docs_remain_ls[i]))

        # 生成无引证回答
        has_no_reference_index = [i for i in range(num) if i not in set(has_reference_index)]
        inputs3 = [inputs[i] for i in has_no_reference_index]
        prompts_without_refs, answers_without_refs = self.batch_query_llm1(
            inputs=inputs3,
            desc='answer w/o references(model1)'
        )

        answers_final = [''] * num
        for i, j in enumerate(has_reference_index):
            answers_final[j] = answers_with_refs2[i]

        for i, j in enumerate(has_no_reference_index):
            assert answers_final[j] == ''
            answers_final[j] = answers_without_refs[i]
        
        # 整合回答
        results = []
        cnt = 0
        for i in range(num):
            result = {
                "input": inputs[i],
                "label": self.data[i]['label'],
                "answer": answers_final[i]
            }
            if i in has_reference_index:
                result['with_reference'] = True
                result['references'] = docs_str_ls[cnt]
                result['origin_answer'] = answers_with_refs[cnt]
                result['docs_remain'] = docs_remain_ls[cnt]
                result['prompt'] = prompts_with_refs[cnt]
                cnt += 1
            else:
                result['with_reference'] = False
            valid, correct, predict_label = self._metric(result['answer'], result['label'])
            result['predict_label'] = predict_label
            result['valid'] = valid
            result['correct'] = correct
            results.append(result)
        overall_info = self.compute_overall(results)
        output = {
            "evaluator information": self.evaluate_info(),
            "overall": overall_info,
            "results": results
        }
        output_path = self.output_path.format(with_refs='with_reference')
        self.save_output(output, output_path)

        print(f'Output saved at {output_path}!')

    def run_without_refs(self):

        # 生成无引证回答
        inputs = [item['input'] for item in self.data]
        prompts, answers = self.batch_query_llm1(inputs=inputs, desc=f'answer w/o references')

        num = len(inputs)
        
        results_without_refs = []
        for i in range(num):
            result = {
                "input": inputs[i],
                "answer": answers[i],
                "label": self.data[i]['label'],
                "with_reference": False
            }
            valid, correct, predict_label = self._metric(result['answer'], result['label'])
            result['predict_label'] = predict_label
            result['valid'] = valid
            result['correct'] = correct
            results_without_refs.append(result)
        overall_info = self.compute_overall(results_without_refs)
        output = {
            "evaluator information": self.evaluate_info(),
            "overall": overall_info,
            "results": results_without_refs 
        }
        output_path = self.output_path.format(with_refs='without_reference')
        self.save_output(output, output_path)

        print(f'Output saved at {output_path}!')


    def _metric(self, answer: str, label: str):
        """标签预测准确性评估"""
        pattern = re.compile(r"\[结论\](.*?)\[分析过程\]", re.DOTALL)
        match = pattern.search(answer)

        valid = False
        correct = False
        predict_label = ''
        if match:
            predict_label = match.group(1).strip()
            for l in ['是谣言', '不是谣言', '与健康信息不相关']:
                if l == predict_label:
                    valid = True
                    correct = (l == label)
                    break
        return valid, correct, predict_label
    
    def compute_overall(self, results: List[dict]) -> dict:
        total_num = len(results)
        vaild_num = sum([r['valid'] for r in results])
        correct_num = sum([r['correct'] for r in results if r['valid']])
        return {
            "total_num": total_num,
            "valid_num": vaild_num,
            "correct_num": correct_num,
            "accuracy": correct_num / vaild_num if vaild_num > 0 else None
        }
    
    def save_output(self, output: dict, output_path: str) -> None:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
