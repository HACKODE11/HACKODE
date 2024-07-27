import os

import torch

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()
    config.model_name = "starchat2-15b"
    config.result_prefix = 'results/individual_starchat2_15b'

    config.tokenizer_paths=["../LLM_Models/starchat2-15b/"]
    config.model_paths=["../LLM_Models/starchat2-15b/"]

    config.data_type = torch.bfloat16
    config.model_kwargs = [{
        'device_map': 'auto'}]

    return config
