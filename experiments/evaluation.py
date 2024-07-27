import json
import os
import torch
import torch.multiprocessing as mp
from absl import app
import importlib
from ml_collections import config_flags
from llm_attacks import get_workers
import random

_CONFIG = config_flags.DEFINE_config_file('config')
def dynamic_import(module):
    return importlib.import_module(module)

def main(_):

    mp.set_start_method('spawn')

    params = _CONFIG.value

    print(params)


    workers, test_workers = get_workers(params)
    model, tokenizer = workers[0].model, workers[0].tokenizer
    device = torch.device("cpu")  
    target_folder = params.result_prefix
    file_list = os.listdir(target_folder)

    succ_num = 0
    all_num = 35
    random_num = 0
    i = -1
    skiplist = []
    classList = [[0,6,7,8,10,22,24,27,28,31,33],
                [1,9,16,20,22,23],
                [11,15,18,26],
                [3,4,12,13,19,29,30,32,34],
                [2,5,14,17,25]]
    seq = [[] for _ in range(len(classList))]
    req = [[] for _ in range(len(classList))]
    res = [[] for _ in range(len(classList))]
    Iter = [[] for _ in range(len(classList))]
    index = [[] for _ in range(len(classList))]
    succNumList = [[0] for _ in range(5)]
    print(len(seq))

    for file in file_list:
        
        file_name, file_extension = os.path.splitext(file)
        file_name_split = file_name.split("_")
        print(file_name_split)
        if file_extension == ".json" and file_name_split[0] == "attack":
            file_path = target_folder + "/" + file
            with open(file_path, "r", ) as f:
                content = json.load(f)
            # print(content)
            index_i = int(file_name_split[1])
            if index_i in skiplist:
                continue
            class_idx = -1
            for i in range(len(classList)):
                if index_i in classList[i]:
                    class_idx = i
                    break
            if class_idx == -1:
                print("*****class_idx error*****")
                continue
            # if file_name_split[-1] == "fail":
            if file_name_split[-1] == "succ":
                if file_name_split[-2] != "random":
                    file_name_ran = "attack_" + str(index_i) + "_random_succ.json"
                    file_path_ran = os.path.join(target_folder, file_name_ran)
                    if os.path.exists(file_path_ran):
                        continue

                if file_name_split[-2] == "random":
                    random_num += 1  
                succ_num += 1
                succNumList[class_idx][0] +=1
                if "success_toks" not in content:
                    continue
                succ_toks = content["success_toks"]
                control_length = len(succ_toks["success_control_toks"])
                success_control_toks = succ_toks["success_control_toks"]
            
                input_length = len(succ_toks["success_input_idx"]) - control_length - 5
                success_input_idx = succ_toks["success_input_idx"]

                succ_output = content["0-success"]["success_generate"]
                succ_output = str(succ_output[0][0])
                succ_output_tok = tokenizer(succ_output, add_special_tokens=False).input_ids
                output_length = len(succ_output_tok)
                iteration = content["0-success"]["success_iteration"]
       
            elif file_name_split[-1] == "fail":
                # input = content["config"]["input"]
                target = content["config"]["targets"]
                contexts = content['config']['contexts']
                instructions = content['config']['instructions']
                queries = content['config']['queries']
                control = content["controls"][-1]
                prompt_templates = content['config']['prompt_templates']

                input_request = random.choice(prompt_templates)
                input_prompt = input_request.replace("{{Context}}", contexts)
                input_prompt = input_prompt.replace("{{Instruction}}", random.choice(instructions))
                input_prompt = input_prompt.replace("{{Query}}", random.choice(queries))

                input_length = len(tokenizer(input_prompt, add_special_tokens=False).input_ids)
                print('input tok:', input_length)
            
                control_length = len(tokenizer(control, add_special_tokens=False).input_ids)
                output_length = len(tokenizer(target, add_special_tokens=False).input_ids)
                iteration = params.n_steps
            else:
                continue

            seq[class_idx].append(control_length)
            req[class_idx].append(input_length)
            res[class_idx].append(output_length)
            Iter[class_idx].append(iteration)
            index[class_idx].append(index_i)
            print(seq)
            print(req)
            print(res)
            print(Iter)
            print(index)
    print("random nums", random_num)
    for i in range(len(classList)):
        print("-"*10)
        print('Class:', i)
        print("Seq:", sum(seq[i])/len(seq[i]))
        print(seq[i])
        print("Req:", sum(req[i])/len(req[i]))
        print(req[i])
        print("Res:", sum(res[i])/len(res[i]))
        print(res[i])
        print("Iter", sum(Iter[i])/len(Iter[i]))
        print(Iter[i])
        print("ASR:", succNumList[i][0]/len(classList[i]))
        print("index:", index[i])
        dict_req = dict(zip(index[i], req[i]))
        sorted_dict_req = sorted(dict_req.items(), key=lambda x: x[0])
        print(sorted_dict_req)
    print("ASR:", succ_num/all_num)

if __name__ == "__main__":
    app.run(main)