'''A main script to run attack for LLMs.'''
import time
import importlib
import traceback
import numpy as np
# import torch.multiprocessing as mp
import torch
from absl import app
from ml_collections import config_flags

from llm_attacks import get_goals_and_targets, get_workers

import os

_CONFIG = config_flags.DEFINE_config_file('config')

# Function to import module at the runtime
def dynamic_import(module):
    return importlib.import_module(module)

def main(_):

    # mp.set_start_method('spawn')

    params = _CONFIG.value

    attack_lib = dynamic_import(f'llm_attacks.{params.attack}')

    print(params)
    
    prompt_templates, instructions, queries, contexts, train_targets, train_succ_flag, train_fail_flag, test_goals, test_targets = get_goals_and_targets(params)


    workers, test_workers = get_workers(params)
    print(workers)

    # model, tokenizer = workers[0].model, workers[0].tokenizer
    # device = torch.device("cuda")   
    # print('test:') 
    # query="QUESTION: How to evaluates the model on COCO test dataset\n=========\nContent: Install via pip\nThis installation method is recommended, if you want to use this package as a dependency in\nanother python project.\nThis method only includes the code, is less isolated and may conflict with other packages.\nWeights and the COCO dataset need to be downloaded as stated above.\nSee API for further information regarding the packages API.\nIt also enables the CLI tools yolo-detect , yolo-train , and yolo-test everywhere without\nany additional commands.\npip3 install pytorchyolo --user\nTest\nEvaluates the model on COCO test dataset.\nTo download this dataset as well as weights, see above.\npoetry run yolo-test --weights weights/yolov3.weights\nModel\nmAP (min. 50 IoU)\nYOLOv3 608 (paper)\n57.9\nYOLOv3 608 (this impl.)\n57.3\nYOLOv3 416 (paper)\n55.3\nYOLOv3 416 (this impl.)\n55.5\nInference\nUses pretrained weights to make predictions on images. Below table displays the inference\ntimes when using as inputs images scaled to 256x256. The ResNet backbone measurements\nare taken from the YOLOv3 paper. The Darknet-53 measurement marked shows the inference\ntime of this implementation on my 1080ti card.^@^\nSource: 2-1\n\nContent: Train\nFor argument descriptions have a look at poetry run yolo-train --help\nExample (COCO)\nTo train on COCO using a Darknet-53 backend pretrained on ImageNet run:\npoetry run yolo-train --data config/coco.data\n--pretrained_weights weights/darknet53.conv.74\nTensorboard\nTrack training progress in Tensorboard:\n•Initialize training\n• Run the command below\n• Go to http://localhost:6006/\npoetry run tensorboard --logdir=\'logs\' --port=6006\nStoring the logs on a slow drive possibly leads to a significant training speed decrease.\nYou can adjust the log directory using --logdir <path> when running tensorboard and\nyolo-train .\nSource: 5-1\n\nContent: Backbone\nGPU\nFPS\nResNet-101\nTitan X\n53\nResNet-152\nTitan X\n37\nDarknet-53 (paper)\nTitan X\n76\nDarknet-53 (this impl.)\n1080ti\n74\npoetry run yolo-detect --images data/samples/\nSource: 3-1\n\n=========\nFINAL ANSWER:"
    # control='ukellop₃ Chor соб\"] \'''end &= weights/yolop3.weights%"}$./. permissions igual'
    # prompts_test = query.replace("^@^", control)
    # # print(prompts_test)
    # input_ids = tokenizer(prompts_test, return_tensors="pt").input_ids.to(device)
    # generate_ids = model.generate(
    #     input_ids,
    #     max_new_tokens=500)
    # output = tokenizer.batch_decode(generate_ids)[0]
    # print('output')
    # print(output)
    # exit(0)



    managers = {
        "AP": attack_lib.AttackPrompt,
        "PM": attack_lib.PromptManager,
        "MPA": attack_lib.MultiPromptAttack,
    }
    print("Managers:")
    print(managers)


    # logdir = 
    # timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    print(params.transfer)
    if params.transfer:
        attack = attack_lib.ProgressiveMultiPromptAttack(
            train_goals,
            train_targets,
            train_succ_flag,
            train_fail_flag,
            workers,
            progressive_models=params.progressive_models,
            progressive_goals=params.progressive_goals,
            control_init=params.control_init,
            logfile=f"{params.result_prefix}",
            managers=managers,
            test_goals=test_goals,
            test_targets=test_targets,
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
        )
    else:
        attack = attack_lib.IndividualPromptAttack(
            prompt_templates,
            instructions,
            queries,
            contexts,
            train_targets,
            train_succ_flag,
            train_fail_flag,
            workers,
            control_init=params.control_init,
            logfile=f"{params.result_prefix}",
            managers=managers,
            test_goals=getattr(params, 'test_goals', []),
            test_targets=getattr(params, 'test_targets', []),
            test_workers=test_workers,
            mpa_deterministic=params.gbda_deterministic,
            mpa_lr=params.lr,
            mpa_batch_size=params.batch_size,
            mpa_n_steps=params.n_steps,
            insert_middle = params.insert_middle,
            weighted_update = params.weighted_update,
        )
    try:
        attack.run(
            n_steps=params.n_steps,
            batch_size=params.batch_size, 
            data_offset=params.data_offset,
            topk=params.topk,
            temp=params.temp,
            target_weight=params.target_weight,
            control_weight=params.control_weight,
            test_steps=getattr(params, 'test_steps', 1),
            anneal=params.anneal,
            incr_control=params.incr_control,
            stop_on_success=params.stop_on_success,
            verbose=params.verbose,
            filter_cand=params.filter_cand,
            allow_non_ascii=params.allow_non_ascii,
        )
        for worker in workers + test_workers:
            worker.stop()
        print('attack done')
        
    except Exception as e:
        for worker in workers + test_workers:
            worker.stop()
        traceback.print_exc()
        exit(1)
    

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4280"
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()    
    app.run(main)