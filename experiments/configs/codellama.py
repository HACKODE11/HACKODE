import os

import torch

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()
    config.model_name = "codellama"
    config.result_prefix = 'results/individual_codellama'

    config.tokenizer_paths=["../LLM_Models/codellama-7b-instruct-hf/"]
    config.model_paths=["../LLM_Models/codellama-7b-instruct-hf/"]
    config.conversation_templates=['llama-2']
    config.data_type = torch.float16
    config.model_kwargs = [{
        'device_map': 'auto'}]

    return config
