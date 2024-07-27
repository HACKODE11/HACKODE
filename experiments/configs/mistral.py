import os

import torch

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()
    config.model_name = "mistral"
    config.result_prefix = 'results/individual_mistral_7b'

    config.tokenizer_paths=["../LLM_Models/Mistral-7B-Instruct-v0.2/"]
    config.model_paths=["../LLM_Models/Mistral-7B-Instruct-v0.2/"]
    config.conversation_templates=['llama-2']
    config.data_type = torch.float16
    config.model_kwargs = [{
        'device_map': 'auto'}]

    return config
