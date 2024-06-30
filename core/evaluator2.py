import re
import json
import copy
import time
import requests
import datetime
from typing import Literal
from multiprocessing import Pool

import numpy as np
from loguru import logger
from tqdm import tqdm
from sklearn.metrics import f1_score
from tenacity import retry, stop_after_attempt, wait_random_exponential


PROMPT_RELEVANCE = """你是一个健康信息问答专家，下面会给出用户输入的健康信息以及针对该健康信息的谣言分析回答，请给出回答与用户输入之间的相关性得分（分值范围为 0-10 分）：

用户输入：{input}

回答：{answer}

请重点考虑以下评估指标进行评分：
1. 信息匹配度：回答是否针对输入的健康信息进行展开的。
2. 谣言识别与分析：回答是否重点分析和回答输入的健康信息是否为谣言这一问题。

请严格按照如下格式输出相关性得分：
{{
	"相关性得分": "X"
}}"""

PROMPT_RELIABILITY = """你是一个健康信息问答专家，下面会给出用户输入的健康信息以及针对该健康信息的谣言分析回答，请你给出该谣言分析回答的可信度得分（分值范围为 0-10 分）：

用户输入：{input}

回答：{answer}

请重点考虑以下评估指标进行评分：
1. 准确性：回答是否提供了准确的事实和信息。
2. 权威性：回答是否基于科学的证据和研究，并表明了信息来源。
3. 合理性：[分析过程]的论述和推理过程是否合理，[分析过程]是否支持[结论]。
4. 非误导性：回答是否避免了使用可能产生新误解或混淆的语言。

请严格按照如下格式输出可信度得分：
{{
	"可信度得分": "X"
}}"""

PROMPT_RICHNESS = """你是一个健康信息问答专家，下面会给出用户输入的健康信息以及针对该健康信息的谣言分析回答，请你给出该谣言分析回答的丰富度得分（分值范围为 0-10 分）：

用户输入：{input}

回答：{answer}

请重点考虑以下评估指标进行评分：
1. 多样性: 回答中是否提供了多个角度、观点或者信息来源，可以通过检查回答中的论点、观点、例子或者来源的数量和多样性来评估。
2. 完整性: 回答是否全面，是否覆盖了用户输入中提及的所有关键点或方面，并进行了相关分析。
3. 创造性: 回答是否提供了新颖的观点、想法或者分析方案。

请严格按照如下格式输出丰富度得分：
{{
	"丰富度得分": "X"
}}"""


def get_f1(
        data: dict, 
        average: Literal["macro", "micro", "weighted"] = "macro"
):
    label_dict = {"是谣言": 0,
                  "不是谣言": 1,
                  "与健康信息不相关": 2}

    true_labels = []
    predicted_labels = []
    for item in data['results']:
        if item['predict_label'] not in label_dict.keys():
            continue 
        true_labels.append(label_dict[item['label']])
        predicted_labels.append(label_dict[item['predict_label']])
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # 计算多类别的F1-score
    f1_value = f1_score(true_labels, predicted_labels, average=average)
    
    return f1_value


@retry(wait=wait_random_exponential(min=30, max=60), stop=stop_after_attempt(5), reraise=True)
def _query_gpt(query):
    data = {
        "model": "gpt-4-1106-preview", #填入想要使用的模型
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.1
    }
    key = ''  # token
    url = ''  # opanai api
    headers = {
            'Authorization': 'Bearer {}'.format(key),
            'Content-Type': 'application/json',
        }
    response = requests.request("POST", 
                                url, 
                                headers=headers, 
                                data=json.dumps(data),
                                timeout=300)

    res = response.json()
    real_res = res["choices"][0]["message"]["content"]

    return real_res


def query_gpt(query):
    try:
        response = _query_gpt(query)
    except Exception as e:
        logger.warning(repr(e))
        response = ''
    return response


def get_score_value(text, key):
    try:
        score = float(json.loads(text[text.find('{'):text.find('}')+1])[key])
    except:
        score = -1
    return score


def get_3r_score(data: dict, data_name: str, process_num: int = 5):

    pattern = re.compile(r"(?:\[[0-9A-Z]{8}\])+")

    relevance_data = []
    reliability_data = []
    richness_data = []
    for item in data:
        answer = item['answer']
        matches = pattern.finditer(answer)
        targets = [i.group() for i in matches]
        targets = list(set(targets))
        for t in targets:
            answer = answer.replace(t, '')

        relevance_data.append(
            PROMPT_RELEVANCE.format(
                input=item['input'],
                answer=answer
            )
        )
        reliability_data.append(
            PROMPT_RELIABILITY.format(
                input=item['input'],
                answer=answer
            )
        )
        richness_data.append(
            PROMPT_RICHNESS.format(
                input=item['input'],
                answer=answer
            )
        )

    if process_num != 1:
        with Pool(process_num) as pool:
            relevance_scores = list(tqdm(
                pool.imap(
                    query_gpt,
                    relevance_data
                ),
                total = len(relevance_data),
                desc = 'relevance score'
            ))
    else:
        relevance_scores = []
        for item in tqdm(relevance_data, desc='relevance score'):
            relevance_scores.append(query_gpt(item))

    time.sleep(600)

    if process_num != 1:
        with Pool(process_num) as pool:
            reliability_scores = list(tqdm(
                pool.imap(
                    query_gpt,
                    reliability_data
                ),
                total = len(reliability_data),
                desc = 'reliability score'
            ))
    else:
        reliability_scores = []
        for item in tqdm(reliability_data, desc='reliability score'):
            reliability_scores.append(query_gpt(item))

    time.sleep(600)
    
    if process_num != 1:
        with Pool(process_num) as pool:
            richness_scores = list(tqdm(
                pool.imap(
                    query_gpt,
                    richness_data
                ),
                total = len(richness_data),
                desc = 'richness score'
            ))
    else:
        richness_scores = []
        for item in tqdm(richness_data, desc='richness score'):
            richness_scores.append(query_gpt(item))

    new_data = {
        "info": {},
        "results": []
    }
    for i, item in enumerate(data):
        new_item = copy.deepcopy(item)
        new_item['relevance_score_str'] = relevance_scores[i]
        new_item['reliability_score_str'] = reliability_scores[i]
        new_item['richness_score_str'] = richness_scores[i]
        new_item['relevance_score'] = get_score_value(relevance_scores[i], "相关性得分")
        new_item['reliability_score'] = get_score_value(reliability_scores[i], "可信度得分")
        new_item['richness_score'] = get_score_value(richness_scores[i], "丰富度得分")
        new_data['results'].append(new_item)
    
    relevance_scores_ls = [
        item['relevance_score'] for item in new_data['results'] if item['relevance_score'] != -1
    ]
    new_data['info']['relevance_score_avg'] = sum(relevance_scores_ls) / len(relevance_scores_ls)
    reliability_scores_ls = [
        item['reliability_score'] for item in new_data['results'] if item['reliability_score'] != -1
    ]
    new_data['info']['reliability_score_avg'] = sum(reliability_scores_ls)/len(reliability_scores_ls)
    richness_scores_ls = [
        item['richness_score'] for item in new_data['results'] if item['richness_score'] != -1
    ]
    new_data['info']['richness_score_avg'] = sum(richness_scores_ls)/len(richness_scores_ls)
    new_data['info']['data_name_from'] = data_name

    return new_data


if __name__ == '__main__':
    base_path = '../outputs/evaluation_result/'
    data_name = ''

    with open(base_path + data_name, 'r') as f:
        data = json.load(f)

    r = get_3r_score(data['results'], data_name=data_name, process_num=4)

    with open(base_path + f'3r_score_{data_name}.json', 'w') as f:
        json.dump(r, f, ensure_ascii=False, indent=4)
    print(base_path + f'3r_score_{data_name}.json')

