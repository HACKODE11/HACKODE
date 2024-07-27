import os

import torch

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()
    config.model_name = "llama2"
    config.result_prefix = 'results/individual_llama2'

    config.tokenizer_paths=["../LLM_Models/llama-2-7b-chat-hf/"]
    config.model_paths=["../LLM_Models/llama-2-7b-chat-hf/"]
    config.conversation_templates=['llama-2']
    config.data_type = torch.float16
    config.model_kwargs = [{
        'device_map': 'auto'}]

    return config
