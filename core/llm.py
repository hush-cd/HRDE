import copy
import json
import os
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_random_exponential

from utils import load_yaml_conf


# BASE_PATH = os.path.abspath(os.path.dirname(__file__))
# PROMPT_DIR = os.path.join(os.path.dirname(BASE_PATH), 'prompts/generate')
# LLM_CONF_PATH = os.path.join(os.path.dirname(BASE_PATH), 'configs/llm.yaml')
PROMPT_DIR = '../prompts/generate'
LLM_CONF_PATH = '../configs/llm.yaml'
TIMEOUT = 45


class BaseLLM(ABC):
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        top_k: int = 5,
        prompt_dir: str = PROMPT_DIR,
        **more_params
    ):
        self.params = {
            'model_name': model_name if model_name else self.__class__.__name__,
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'top_k': top_k,
            **more_params
        }
        self.prompt_dir = prompt_dir
        # Load information of current LLM
        self.conf = load_yaml_conf(LLM_CONF_PATH)[self.__class__.__name__]

    @abstractmethod
    def _request(self, query: str) -> str:
        """Without further processing the response of the request; simply return the string."""
        return ''

    def _read_prompt_template(self, filename: str) -> str:
        path = os.path.join(self.prompt_dir, filename)
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
        else:
            logger.error(f'Prompt template not found at {path}')
            return ''

    def update_params(self, inplace: bool = True, **params):
        """Update parameters either in-place or create a new object with updated parameters.
        """
        if inplace:
            self.params.update(params)
            return self
        else:
            new_obj = copy.deepcopy(self)
            new_obj.params.update(params)
            return new_obj

    def safe_request(self, query: str) -> str:
        """Safely make a request to the language model, handling exceptions."""
        try:
            response = self._request(query)
        except Exception as e:
            import traceback
            print(traceback.print_exc())
            logger.warning(repr(e))
            response = ''
        return response

    # ─── Prompt Engineering ───────────────────────────────────────────────
    def refute_rumor(self, input_info: Tuple[str, list]):
        """
        Refuting process
        input: User-entered content that needs to be refuted
        references: References based on the refuting content recall. If None, provide an answer without reference; otherwise, provide an answer with reference.
        """
        input, references = input_info

        refute_rumor_template = self._read_prompt_template('refute_rumor.txt')
        refute_rumor_with_reference_template = self._read_prompt_template('refute_rumor_with_reference.txt')

        if references:
            reference_str = '\n'.join(references)
            prompt = refute_rumor_with_reference_template.format(input=input,
                                                                 references=reference_str)
        else:
            prompt = refute_rumor_template.format(input=input)

        response = self.safe_request(prompt)

        return (prompt, response)
    
    def process_prompt(self, input_info: Tuple[str, list]):
        """
        Refuting process (for streaming generation)
        input: User-entered content that needs to be refuted
        references: References based on the refuting content recall. If None, provide an answer without reference; otherwise, provide an answer with reference.
        """
        input, references = input_info

        refute_rumor_template = self._read_prompt_template('refute_rumor.txt')
        refute_rumor_with_reference_template = self._read_prompt_template('refute_rumor_with_reference.txt')

        if references:
            reference_str = '\n'.join(references)
            prompt = refute_rumor_with_reference_template.format(input=input,
                                                                 references=reference_str)
        else:
            prompt = refute_rumor_template.format(input=input)
        return prompt
    
    @abstractmethod
    def stream_request(self, prompt):
        pass


    
class VllmModel(BaseLLM):
    def _base_prompt_template(self) -> str:
        return "{query}"

    def _request(self, query: str) -> str:
        url = self.conf['url']

        template = self._base_prompt_template()
        query = template.format(query=query)
        payload = json.dumps({
            "prompt": query,
            "temperature": self.params['temperature'],
            "max_tokens": self.params['max_new_tokens'],
            "n": 1,
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        })
        headers = {
            'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        
        res = res.json()['text'][0].replace(query, '')  # VLLM will automatically append the query to the response, so here we remove it.
        res = self._post_process_response(res)
        return res
    
    def _post_process_response(self, response: str) -> str:
        return response
    
    def stream_request(self, prompt):
        url = self.conf['url']

        template = self._base_prompt_template()
        query = template.format(query=prompt)
        payload = json.dumps({
            "prompt": query,
            "temperature": self.params['temperature'],
            "max_tokens": self.params['max_new_tokens'],
            "n": 1,
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
            "stream": True
        })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload, stream=True)
        for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode("utf-8"))
                output = data["text"]
                yield output


class Qwen_14B_Chat(VllmModel):
    def _base_prompt_template(self) -> str:
        template = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n""" \
                   """{query}<|im_end|>\n<|im_start|>assistant\n"""
        return template

    def _request(self, query: str) -> str:
        url = self.conf['url']

        template = self._base_prompt_template()
        query = template.format(query=query)
        payload = json.dumps({
            "prompt": query,
            "temperature": self.params['temperature'],
            "max_tokens": self.params['max_new_tokens'],
            "n": 1,
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
            "stop": ["<|endoftext|>", "<|im_end|>"],
        })
        headers = {
            'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        print(res.text)
        res = res.json()['text'][0].replace(query, '')  # VLLM will automatically append the query to the response, so here we remove it.
        res = self._post_process_response(res)
        return res


class Qwen_4B_Chat(VllmModel):
    def _base_prompt_template(self) -> str:
        template = """<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n""" \
                   """{query}<|im_end|>\n<|im_start|>assistant\n"""
        return template

    def _request(self, query: str) -> str:
        url = self.conf['url']

        template = self._base_prompt_template()
        query = template.format(query=query)
        payload = json.dumps({
            "prompt": query,
            "temperature": self.params['temperature'],
            "max_tokens": self.params['max_new_tokens'],
            "n": 1,
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
            "stop": ["<|endoftext|>", "<|im_end|>"],
        })
        headers = {
            'Content-Type': 'application/json'
        }
        res = requests.request("POST", url, headers=headers, data=payload)
        res = res.json()['text'][0].replace(query, '')  # VLLM will automatically append the query to the response, so here we remove it.
        res = self._post_process_response(res)
        return res
