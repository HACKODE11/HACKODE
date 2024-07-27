import gc
import json
import math
import os
import queue
import random
import threading
import time
from copy import deepcopy
import traceback
from typing import Optional, Any
from llm_attacks.base.utils import find_token
import numpy as np
import pandas as pd
import torch
import pickle
import torch.multiprocessing as mp

import torch.nn as nn
import torch.nn.functional as F
from fastchat.model import get_conversation_template
from transformers import (AutoModelForCausalLM,   AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM,
                          LlamaForCausalLM, BitsAndBytesConfig, MistralForCausalLM, Starcoder2ForCausalLM)
from transformers import PreTrainedModel
from transformers import pipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import warnings
warnings.filterwarnings("ignore")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_embedding_layer(model):
    # if isinstance(model, PeftModelForCausalLM):
    #     model = model.base_model.model
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte
    elif isinstance(model, (LlamaForCausalLM, MistralForCausalLM, Starcoder2ForCausalLM)):
        return model.model.embed_tokens
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

# from peft import PeftModelForCausalLM

def get_embedding_matrix(model):
    # if isinstance(model, PeftModelForCausalLM):
    #     model = model.base_model.model
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte.weight
    elif isinstance(model, (LlamaForCausalLM, MistralForCausalLM, Starcoder2ForCausalLM)):
        return model.model.embed_tokens.weight
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in.weight
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def get_embeddings(model, input_ids):
    # if isinstance(model, PeftModelForCausalLM):
    #     model = model.base_model.model
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte(input_ids).half()
    elif isinstance(model, (LlamaForCausalLM, MistralForCausalLM, Starcoder2ForCausalLM)):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in(input_ids).half()
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_nonascii_toks(tokenizer, device='cpu'):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)
    
    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)
    
    return torch.tensor(ascii_toks, device=device)

class AttackPrompt(object):
    """
    A class used to generate an attack prompt. 
    """
    
    def __init__(self,
        prompt_templates,
        instructions,
        queries,
        context,
        target,
        succ_flags,
        fail_flags,
        temp,
        tokenizer,
        conv_template,
        control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        insert_middle = False,
        weighted_update = -1,
        # test_prefixes=["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"],
        *args, **kwargs
    ):
        """
        Initializes the AttackPrompt object with the provided parameters.

        Parameters
        ----------
        goal : str
            The intended goal of the attack
        target : str
            A sample of taget attack
        tokenizer : Transformer Tokenizer
            The tokenizer used to convert text into tokens
        conv_template : Template
            The conversation template used for the attack
        control_init : str, optional
            A string used to control the attack (default is "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ")
        test_prefixes : list, optional
            A list of prefixes to test the attack (default is ["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"])
        """
        
        self.prompt_templates = prompt_templates
        self.instructions = instructions
        self.queries = queries
        self.context = context
        self.sample_target = target
        self.succ_flags = succ_flags
        self.fail_flags = fail_flags
        self.control = control_init
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.weighted_update = weighted_update
        self.temp=temp

        # insert succ flag
        if insert_middle:
            self.control = self.update_control()
            
        print('initial control:', self.control)
        # self.test_prefixes = test_prefixes

        self.conv_template.messages = []

        self.test_new_toks = len(self.tokenizer(self.sample_target).input_ids) + 50 # buffer
        for succ, fail in zip(self.succ_flags, self.fail_flags):
            self.test_new_toks = max(self.test_new_toks, len(self.tokenizer(succ).input_ids))
            self.test_new_toks = max(self.test_new_toks, len(self.tokenizer(fail).input_ids))

        self._update_ids()
       

    def _update_ids(self, is_random = False):
        if is_random:
            self.goal = random.choice(self.prompt_templates)
            self.goal = self.goal.replace("{{Context}}", self.context)
            self.goal = self.goal.replace("{{Instruction}}", random.choice(self.instructions))
            self.goal = self.goal.replace("{{Query}}", random.choice(self.queries))
        else:
            self.goal = self.prompt_templates[0]
            self.goal = self.goal.replace("{{Context}}", self.context)
            self.goal = self.goal.replace("{{Instruction}}", self.instructions[0])
            self.goal = self.goal.replace("{{Query}}", self.queries[0])

        if "^@^" in self.goal:   # Insert Position
            self.attack_goal = self.goal.replace("^@^", self.control)
        else:
            self.attack_goal = self.goal + " " + self.control

        self.conv_template.append_message(self.conv_template.roles[0], f"{self.attack_goal}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.sample_target}")
        prompt = self.conv_template.get_prompt()
        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids
        print('self.conv_template.name', self.conv_template.name)
        if 'llama' in self.conv_template.name or 'one_shot' in self.conv_template.name :
        # if True:
            self.conv_template.messages = []

            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt().replace('\n', '\\n')).input_ids
            self._user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(f"{self.attack_goal}")
            toks = self.tokenizer(self.conv_template.get_prompt().replace('\n', '\\n')).input_ids
            control_pos = find_token(self.tokenizer, toks, self.control)
            
            assert len(control_pos)>0, "The control should be inserted in the origin sentence."
            control_start, control_end = control_pos[0]
            self._goal_slice = [
                slice(self._user_role_slice.stop, control_start),
                slice(control_end + 1, max(self._user_role_slice.stop, len(toks)))]
        

            self._control_slice = slice(control_start, control_end)

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt().replace('\n', '\\n')).input_ids
            
            self._assistant_role_slice = slice(self._goal_slice[1].stop, len(toks))
    
            self.conv_template.update_last_message(f"{self.sample_target}")
            toks = self.tokenizer(self.conv_template.get_prompt().replace('\n', '\\n')).input_ids
            
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-2) # todo check what is slice
            self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-3)

            self._keywords_slices = []
            for succ in self.succ_flags:
                self._keywords_slices += find_token(self.tokenizer, toks[self._target_slice], succ)
            assert len(self._keywords_slices)>0, "The keywords should be found in the generated sentence."

        else:
            python_tokenizer = False or self.conv_template.name == 'oasst_pythia'
            try:
                encoding.char_to_token(len(prompt)-1)
            except:
                python_tokenizer = True
            if python_tokenizer:
                # This is specific to the vicuna and pythia tokenizer and conversation prompt.
                # It will not work with other tokenizers or prompts.
                self.conv_template.messages = []

                self.conv_template.append_message(self.conv_template.roles[0], None)
                toks = self.tokenizer(self.conv_template.get_prompt().replace('\n', '\\n')).input_ids
                self._user_role_slice = slice(None, len(toks))

                self.conv_template.update_last_message(f"{self.attack_goal}")

                toks = self.tokenizer(self.conv_template.get_prompt().replace('\n', '\\n')).input_ids
                control_pos = find_token(self.tokenizer, toks, self.control)
                assert len(control_pos)>0, "The control should be inserted in the origin sentence."
                control_start, control_end = control_pos[0]
                assert control_start!=-1, "The control should be inserted in the origin sentence."
                self._goal_slice = [
                slice(self._user_role_slice.stop, control_start),
                slice(control_end + 1, max(self._user_role_slice.stop, len(toks)))]

                self._control_slice = slice(control_start, control_end)

                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks = self.tokenizer(self.conv_template.get_prompt().replace('\n', '\\n')).input_ids
                self._assistant_role_slice = slice(self._goal_slice[1].stop, len(toks))

                self.conv_template.update_last_message(f"{self.sample_target}")
                toks = self.tokenizer(self.conv_template.get_prompt().replace('\n', '\\n')).input_ids
                self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-1)
                self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-2)

                self._keywords_slices = []
                for succ in self.succ_flags:
                
                    self._keywords_slices += find_token(self.tokenizer, toks[self._target_slice], succ)
                assert len(self._keywords_slices)>0, "The keywords should be found in the generated sentence."

            else:
                assert False, "needs debug for modifying the insert position"
                self._system_slice = slice(
                    None, 
                    encoding.char_to_token(len(self.conv_template.system))
                )
                self._user_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0]) + len(self.conv_template.roles[0]) + 1)
                )

                self._goal_slice = [slice(
                        encoding.char_to_token(prompt.find(self.goal)),
                        encoding.char_to_token(prompt.find(self.control))),
                    slice(
                        encoding.char_to_token(prompt.find(self.control) + len(self.control)),
                        encoding.char_to_token(prompt.find(self.goal) + len(self.goal)),
                    )
                ]
                self._control_slice = slice(
                    encoding.char_to_token(prompt.find(self.control)),
                    encoding.char_to_token(prompt.find(self.control) + len(self.control))
                )
                self._assistant_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1)
                )
                self._target_slice = slice(
                    encoding.char_to_token(prompt.find(self.sample_target)),
                    encoding.char_to_token(prompt.find(self.sample_target) + len(self.sample_target))
                )
                self._loss_slice = slice(
                    encoding.char_to_token(prompt.find(self.sample_target)) - 1,
                    encoding.char_to_token(prompt.find(self.sample_target) + len(self.sample_target)) - 1
                )

        self.input_ids = torch.tensor(toks[:self._target_slice.stop], device='cpu')
        self.conv_template.messages = []
        if self.weighted_update != -1:
            self.weights = torch.ones(self._target_slice.stop-self._target_slice.start , dtype=torch.float32) * (1 - self.weighted_update)
            for key in self._keywords_slices:
                self.weights[key[0]: key[1]]  = self.weighted_update
            self.weights = self.weights.to(self.input_ids.device)

    def update_control(self):
        for idx, succ in enumerate(self.succ_flags):
            clean_target = self.sample_target.replace('\n', '\\n')
            succ_ids = self.tokenizer(clean_target).input_ids
            diffs = find_token(self.tokenizer, succ_ids, succ)
            # print('diffs:', diffs)
            assert len(diffs)>0, "Cannot find the position of malicious words!"
            diff_start, diff_end = diffs[0]

            len_diff = diff_end - diff_start +1
            middle = len(self.control) //2
            new_control = self.control[:middle] + self.tokenizer.decode(succ_ids[diff_start:diff_end+1]) + self.control[middle+1:]

            return new_control

    @torch.no_grad()
    def generate(self, model, gen_config=None):
        if gen_config is None:
            gen_config = model.generation_config
            gen_config.max_new_tokens = 1000
        print('model.name', model.name)
        if "codellama" in model.name:
            gen_config.do_sample = False
            # print("do sample set True")
        elif "llama" in model.name:
            gen_config.do_sample = False
            print("do sample set false")
        gen_config.max_new_tokens = gen_config.max_new_tokens + 250
        gen_config.temperature = self.temp
        input_ids = self.input_ids[:self._assistant_role_slice.stop].to(model.device).unsqueeze(0)
        attn_masks = torch.ones_like(input_ids).to(model.device)
       

        output_ids = model.generate(input_ids=input_ids, 
                                    attention_mask=attn_masks, 
                                    generation_config=gen_config,
                                    pad_token_id=self.tokenizer.pad_token_id)[0]
 
        return output_ids[self._assistant_role_slice.stop:]
    
    def generate_str(self, model, gen_config=None):
        return self.tokenizer.decode(self.generate(model, gen_config))
    
    def test(self, model, gen_config=None):
        if gen_config is None:
            gen_config = model.generation_config
            gen_config.max_new_tokens = self.test_new_toks
        gen_config.temperature = self.temp
        gen_str = self.generate_str(model, gen_config).strip()

        jailbroken = not any([fail in gen_str for fail in self.fail_flags])
        jailbroken &= all([succ in gen_str for succ in self.succ_flags])       
        em = self.sample_target in gen_str
        return int(jailbroken), int(em), gen_str

    @torch.no_grad()
    def test_loss(self, model):
   
        logits, ids = self.logits(model, return_ids=True)
        return self.target_loss(logits, ids).mean().item() # 
    
    def grad(self, model):
        
        raise NotImplementedError("Gradient function not yet implemented")
    
    @torch.no_grad()
    def logits_with_generated_string(self, input_ids, model, attention_mask=None):
        gen_config = model.generation_config
        gen_config.max_new_tokens = self.test_new_toks
        gen_config.temperature = self.temp
        if "codellama" in model.name:
            gen_config.do_sample = False
            # print("do sample set True")
        elif 'llama' in model.name:
            gen_config.do_sample=False
            print("do sample set false")
        generate_time = time.time()
        if attention_mask is None:
            
            output = model.generate(input_ids=input_ids[:,:self._target_slice.start], output_scores = True, return_dict_in_generate=True, generation_config=gen_config)
        else:
            output = model.generate(input_ids=input_ids[:,:self._target_slice.start], attention_mask=attention_mask[:,:self._target_slice.start], output_scores = True, return_dict_in_generate=True, generation_config=gen_config)
        generate_time = time.time() - generate_time
        print("???? generate_time: ", generate_time)
        scores = torch.stack(output.scores, dim=1)
        return scores, output.sequences[:, self._target_slice.start:]

    @torch.no_grad()
    def logits(self, model, test_controls=None, return_ids=False, return_str=False):
        pad_tok = -1
        if test_controls is None:
            test_controls = self.control_toks
        if isinstance(test_controls, torch.Tensor):
            if len(test_controls.shape) == 1:
                test_controls = test_controls.unsqueeze(0)
            test_ids = test_controls.to(model.device)
        elif not isinstance(test_controls, list):
            test_controls = [test_controls]
        elif isinstance(test_controls[0], str):
            max_len = self._control_slice.stop - self._control_slice.start
            test_ids = [
                torch.tensor(self.tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
                for control in test_controls
            ]
            pad_tok = 0
            while pad_tok in self.input_ids or any([pad_tok in ids for ids in test_ids]):
                pad_tok += 1
            nested_ids = torch.nested.nested_tensor(test_ids)
            test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
        else:
            raise ValueError(f"test_controls must be a list of strings or a tensor of token ids, got {type(test_controls)}")
        
        if not(test_ids[0].shape[0] == self._control_slice.stop - self._control_slice.start):
            raise ValueError((
                f"test_controls must have shape "
                f"(n, {self._control_slice.stop - self._control_slice.start}), " 
                f"got {test_ids.shape}"
            ))
        
        locs = torch.arange(self._control_slice.start, self._control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device) # shape (1, len(control))
        ids = torch.scatter(  ## insert control into input_ids
            self.input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
            1,
            locs,
            test_ids
        )
        if pad_tok >= 0:
            attn_mask = (ids != pad_tok).type(ids.dtype)
        else:
            attn_mask = None
        if return_ids:
            del locs, test_ids ; gc.collect()
            if return_str:
                logits, ret_str = self.logits_with_generated_string(ids, model, attention_mask=attn_mask)
                return logits, ret_str, ids
            return model(input_ids=ids, attention_mask=attn_mask).logits, ids
        else:
            del locs, test_ids
            if return_str:
                logits, generate_str = self.logits_with_generated_string(ids, model, attention_mask=attn_mask)
                del ids ; gc.collect()
                return logits, generate_str
            else:
                logits = model(input_ids=ids, attention_mask=attn_mask).logits
                del ids ; gc.collect()
                return logits
    
    def target_loss(self, logits, ids, gen_strs=None):
        
        crit = nn.CrossEntropyLoss(reduction='none')
        logits = logits/self.temp
        if gen_strs is not None:
            del ids ; gc.collect()

            len_ids = min(len(logits[0]), len(gen_strs[0])) # todo: control length of sentence.
            new_target_ids = torch.zeros(size=(len(logits), len_ids), device=logits.device, dtype=torch.long)
            for idx, gen_str in enumerate(gen_strs):
                new_gen = self.tokenizer.decode(gen_str[:len_ids])
                if self.fail_flags[0] in new_gen:
                    new_gen = new_gen.replace(self.fail_flags[0], self.succ_flags[0])
                else:
                    new_gen = self.sample_target
                new_ids = self.tokenizer(new_gen, add_special_tokens=False).input_ids[:len_ids]
                new_target_ids[idx] = torch.tensor(new_ids, device=logits.device)
            loss = crit(logits.transpose(1,2), new_target_ids)
        else:
            if self.weighted_update > 0:
                loss = 0
                for keyword in self._keywords_slices:
                    loss_slice = slice(self._target_slice.start+keyword[0]-1, self._target_slice.start+keyword[1]-1)
                    loss += crit(logits[:,loss_slice,:].transpose(1,2), ids[:,loss_slice.start+1:loss_slice.stop+1])
            else:
                loss_slice = slice(self._target_slice.start-1, self._target_slice.stop-1)
                loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,self._target_slice])

        # torch.concatenate(new_target_ids, dim=0)
        return loss
    
    def control_loss(self, logits, ids):
        crit = nn.CrossEntropyLoss(reduction='none')
        loss_slice = slice(self._control_slice.start-1, self._control_slice.stop-1)
        loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,self._control_slice])
        return loss
    
    @property
    def assistant_str(self):
        return self.tokenizer.decode(self.input_ids[self._assistant_role_slice]).strip()
    
    @property
    def assistant_toks(self):
        return self.input_ids[self._assistant_role_slice]

    @property
    def goal_str(self):
        return self.tokenizer.decode(self.input_ids[self._goal_slice[0]] + self.input_ids[self._goal_slice[1]] ).strip()

    @goal_str.setter
    def goal_str(self, goal):
        self.goal = goal
        self._update_ids()
    
    @property
    def goal_toks(self):
        return self.input_ids[self._goal_slice[0]] + self.input_ids[self._goal_slice[1]]
    
    @property
    def target_str(self):
        return self.tokenizer.decode(self.input_ids[self._target_slice]).strip()
    
    @target_str.setter
    def target_str(self, target):
        self.sample_target = target
        self._update_ids()
    
    @property
    def target_toks(self):
        return self.input_ids[self._target_slice]
    
    @property
    def control_str(self):
        return self.tokenizer.decode(self.input_ids[self._control_slice]).strip()
    
    @control_str.setter
    def control_str(self, control):
        self.control = control
        self._update_ids()
    
    @property
    def control_toks(self):
        return self.input_ids[self._control_slice]
    
    @control_toks.setter
    def control_toks(self, control_toks):
        self.control = self.tokenizer.decode(control_toks)
        self._update_ids()
    
    @property
    def prompt(self):
        return self.tokenizer.decode(self.input_ids[self._goal_slice.start:self._control_slice.stop])
    
    @property
    def input_toks(self):
        return self.input_ids

    @property
    def eval_toks(self):
        return self.input_ids[:self._assistant_role_slice.stop]
    
    @property
    def input_str(self):
        return self.tokenizer.decode(self.input_ids)
    
    @property
    def eval_str(self):
        return self.tokenizer.decode(self.input_ids[:self._assistant_role_slice.stop]).replace('<s>','').replace('</s>','')


class PromptManager(object):
    """A class used to manage the prompt during optimization."""
    def __init__(self,
        prompt_templates,
        instructions,
        queries,
        contexts,
        targets,
        succ_flags,
        fail_flags,
        temp,
        tokenizer,
        conv_template,
        control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        # test_prefixes=["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"],
        managers=None,
        insert_middle = False,
        weighted_update = -1,
        *args, **kwargs
    ):
        """
        Initializes the PromptManager object with the provided parameters.

        Parameters
        ----------
        goals : list of str
            The list of intended goals of the attack
        targets : list of str
            The list of targets of the attack
        tokenizer : Transformer Tokenizer
            The tokenizer used to convert text into tokens
        conv_template : Template
            The conversation template used for the attack
        control_init : str, optional
            A string used to control the attack (default is "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
        test_prefixes : list, optional
            A list of prefixes to test the attack (default is ["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"])
        managers : dict, optional
            A dictionary of manager objects, required to create the prompts.
        """

        self.tokenizer = tokenizer

        self._prompts = [  # Individual Attack, one PM only has one goal and one target
            managers['AP'](
                prompt_templates,
                instructions, 
                query,
                context, 
                target,
                succ_flags,
                fail_flags,
                temp,
                tokenizer,
                conv_template,
                control_init,
                insert_middle = insert_middle,
                weighted_update = weighted_update
                # test_prefixes
            )
            for query, context, target in zip(queries, contexts, targets)
        ]

        self._nonascii_toks = get_nonascii_toks(tokenizer, device='cpu')

    def generate(self, model, gen_config=None):
        if gen_config is None:
            gen_config = model.generation_config
            gen_config.max_new_tokens = 16
        if "codellama" in model.name:
            gen_config.do_sample = False
            # print("do sample set True")
        elif "llama" in model.name:
            gen_config.do_sample = False
            print("do sample set false")
        return [prompt.generate(model, gen_config) for prompt in self._prompts]
    
    def generate_str(self, model, gen_config=None):
        return [
            self.tokenizer.decode(output_toks) 
            for output_toks in self.generate(model, gen_config)
        ]
    
    def test(self, model, gen_config=None):
        return [prompt.test(model, gen_config) for prompt in self._prompts]

    def test_loss(self, model):
        return [prompt.test_loss(model) for prompt in self._prompts]
    
    def grad(self, model):
        return sum([prompt.grad(model) for prompt in self._prompts])
    
    def logits(self, model, test_controls=None, return_ids=False):
        vals = [prompt.logits(model, test_controls, return_ids) for prompt in self._prompts]
        if return_ids:
            return [val[0] for val in vals], [val[1] for val in vals]
        else:
            return vals
    
    def target_loss(self, logits, ids):
        return torch.cat(
            [
                prompt.target_loss(logit, id).mean(dim=1).unsqueeze(1)
                for prompt, logit, id in zip(self._prompts, logits, ids)
            ],
            dim=1
        ).mean(dim=1)
    
    def control_loss(self, logits, ids):
        return torch.cat(
            [
                prompt.control_loss(logit, id).mean(dim=1).unsqueeze(1)
                for prompt, logit, id in zip(self._prompts, logits, ids)
            ],
            dim=1
        ).mean(dim=1)
    
    def sample_control(self, *args, **kwargs):

        raise NotImplementedError("Sampling control tokens not yet implemented")

    def __len__(self):
        return len(self._prompts)

    def __getitem__(self, i):
        return self._prompts[i]

    def __iter__(self):
        return iter(self._prompts)
    
    @property
    def control_str(self):
        return self._prompts[0].control_str
    
    @property
    def control_toks(self):
        return self._prompts[0].control_toks

    @control_str.setter
    def control_str(self, control):
        for prompt in self._prompts:
            prompt.control_str = control
    
    @control_toks.setter
    def control_toks(self, control_toks):
        for prompt in self._prompts:
            prompt.control_toks = control_toks

    @property
    def inputs_str(self):
        return [prompt.input_str for prompt in self._prompts]
    
    @property 
    def eval_toks(self):
        return [prompt.eval_toks for prompt in self._prompts]

    @property
    def disallowed_toks(self):
        return self._nonascii_toks

    def random_prompt(self):
        for prompt in self._prompts:
            prompt._update_ids(is_random=True)

class MultiPromptAttack(object): ## the final used attack manager
    """A class used to manage multiple prompt-based attacks."""
    def __init__(self, 
        prompt_templates,
        instructions, 
        queries,
        context,
        targets,
        succ_flags,
        fail_flags,
        temp,
        workers,
        # test_prefixes=["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"],
        control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        logfile=None,
        managers=None,
        test_goals=[],
        test_targets=[],
        test_workers=[],
        insert_middle = False,
        weighted_update = -1,
        *args, **kwargs
    ):
        """
        Initializes the MultiPromptAttack object with the provided parameters.

        Parameters
        ----------
        goals : list of str
            The list of intended goals of the attack
        targets : list of str
            The list of targets of the attack
        workers : list of Worker objects
            The list of workers used in the attack
        control_init : str, optional
            A string used to control the attack (default is "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
        test_prefixes : list, optional
            A list of prefixes to test the attack (default is ["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"])
        logfile : str, optional
            A file to which logs will be written
        managers : dict, optional
            A dictionary of manager objects, required to create the prompts.
        test_goals : list of str, optional
            The list of test goals of the attack
        test_targets : list of str, optional
            The list of test targets of the attack
        test_workers : list of Worker objects, optional
            The list of test workers used in the attack
        """

        self.prompt_templates = prompt_templates
        self.instructions = instructions
        self.queries = queries
        self.context = context
        self.targets = targets
        self.succ_flags = succ_flags
        self.fail_flags = fail_flags
        self.workers = workers
        self.test_goals = test_goals
        self.test_targets = test_targets
        self.test_workers = test_workers
        self.temp=temp
        # self.test_prefixes = test_prefixes
        self.models = [worker.model for worker in workers]
        self.logfile = logfile
        
        if os.path.exists(self.logfile[:-5]+"_succ"+self.logfile[-5:]):
            succ_path = self.logfile[:-5]+"_succ"+self.logfile[-5:]
            with open(succ_path, "r") as f:
                attack_results = json.load(f)
            succ_step_one_control = attack_results["0-success"]["success_cotrol"]
            print('*********attack success! try the random attack*********')
            attack_succ = 1
            control_init = succ_step_one_control
            insert_middle = False

        self.prompts = [
            managers['PM'](
                prompt_templates,
                instructions,
                queries,
                context,
                targets,
                self.succ_flags,
                self.fail_flags,
                temp,
                worker.tokenizer,
                worker.conv_template,
                control_init,
                managers,
                insert_middle=insert_middle,
                weighted_update=weighted_update
            )
            for worker in workers
        ]
        self.managers = managers

    
    @property
    def control_str(self):
        return self.prompts[0].control_str
    
    @control_str.setter
    def control_str(self, control):
        for prompts in self.prompts:
            prompts.control_str = control

    @property
    def input_str(self):
        return [prompt.input_str for prompt in self.prompts]
    
    
    @property
    def control_toks(self):
        return [prompts.control_toks for prompts in self.prompts]
    
    @control_toks.setter
    def control_toks(self, control):
        if len(control) != len(self.prompts):
            raise ValueError("Must provide control tokens for each tokenizer")
        for i in range(len(control)):
            self.prompts[i].control_toks = control[i]
    
    def get_filtered_cands(self, worker_index, control_cand, filter_cand=True, curr_control=None, succ_sentence = None):
        cands, count = [], 0
        worker = self.workers[worker_index]
        # prompt = self.prompts[worker_index].inputs_str[0]
        goal_str = self.prompts[worker_index][0].goal
        # print('control_cand:', control_cand)
        print('succ_sentence:', succ_sentence)
        for i in range(control_cand.shape[0]):
            decoded_str = worker.tokenizer.decode(control_cand[i], skip_special_tokens=True)
          
            if "^@^" in goal_str:   # Insert Position
                attack_goal = goal_str.replace("^@^", decoded_str)
            else:
                attack_goal = goal_str + " " + decoded_str

            if filter_cand:
                toks = worker.tokenizer(attack_goal).input_ids
                str_pos = find_token(worker.tokenizer, toks, decoded_str)
          
                if len(str_pos) >0 :
                    str_start, str_end = str_pos[0] 
                    if decoded_str != curr_control and len(worker.tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]) and (succ_sentence is None or succ_sentence in decoded_str) and (False if "\n" in decoded_str else True):
  
                        cands.append(decoded_str)
                    else:
                        count += 1
                else:
                    count += 1
            else:
                cands.append(decoded_str)
        if len(cands) == 0:
            cands = [curr_control]
        if filter_cand:
            cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
            # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")

        return cands

    def step(self, *args, **kwargs):  # implemented in GCGMultiPromptAttack in gcgAtack
        
        raise NotImplementedError("Attack step function not yet implemented")
    
    def run(self, 
        n_steps=100, 
        batch_size=1024, 
        topk=256, 
        temp=1, 
        allow_non_ascii=True,
        target_weight=None, 
        control_weight=None,
        anneal=True,
        anneal_from=0,
        prev_loss=np.infty,
        stop_on_success=True,
        test_steps=50,
        log_first=False,
        filter_cand=True,
        verbose=True
    ):
        def P(e, e_prime, k):
            T = max(1 - float(k+1)/(n_steps+anneal_from), 1.e-7)
            return True if e_prime < e else math.exp(-(e_prime-e)/T) >= random.random()

        if target_weight is None:
            target_weight_fn = lambda _: 1
        elif isinstance(target_weight, (int, float)):
            target_weight_fn = lambda i: target_weight
        if control_weight is None:
            control_weight_fn = lambda _: 0.1
        elif isinstance(control_weight, (int, float)):
            control_weight_fn = lambda i: control_weight
        steps = 0
        loss = best_loss = 1e6
        best_control = self.control_str
        runtime = 0.
        if self.logfile is not None and log_first:
            model_tests = self.test_all()  # invoke test() method and feed a task into worker

            self.log(anneal_from, 
                     n_steps+anneal_from, 
                     self.control_str, 
                     loss, 
                     runtime, 
                     model_tests, 
                     verbose=verbose)
            self.log_gen_str(self.prompts[0].inputs_str , model_tests[-1])
        if os.path.exists(self.logfile[:-5]+"_succ"+self.logfile[-5:]):
            attack_succ = 1
        else:
            attack_succ = 0
        attack_random_succ = 0
        for i in range(n_steps):
            steps += 1
            start = time.time()

            model_tests=None
            torch.cuda.empty_cache()
            if attack_succ == 0:
                print("close random")
                control, loss = self.step(
                    batch_size=batch_size, 
                    topk=topk, 
                    temp=temp, 
                    allow_non_ascii=allow_non_ascii, 
                    target_weight=target_weight_fn(i), 
                    control_weight=control_weight_fn(i),
                    filter_cand=filter_cand,
                    verbose=verbose,
                    is_random=False
                )
            else:
                print("open random")
                control, loss = self.step(
                    batch_size=batch_size, 
                    topk=topk, 
                    temp=temp, 
                    allow_non_ascii=allow_non_ascii, 
                    target_weight=target_weight_fn(i), 
                    control_weight=control_weight_fn(i),
                    filter_cand=filter_cand,
                    verbose=verbose,
                    is_random=True
                )

            runtime = time.time() - start
            keep_control = True if not anneal else P(prev_loss, loss, i+anneal_from)
            # print('control:', control, 'keep_control:', keep_control)
            if keep_control:  # decide whether use new control or not
                self.control_str = control

            prev_loss = loss
            if loss < best_loss:
                best_loss = loss
                best_control = control
            print('Step:', steps,  'Current Loss:', loss, 'Best Loss:', best_loss)
            
            if self.logfile is not None and (i+1+anneal_from) % test_steps == 0: #todo output more results here
            # if True
                last_control = self.control_str
                self.control_str = best_control
                model_tests = self.test_all()
                self.log(i+1+anneal_from, n_steps+anneal_from, self.control_str, best_loss, runtime, model_tests, verbose=verbose)
                self.log_gen_str(self.prompts[0].inputs_str, model_tests[-1])
                
                self.control_str = last_control
            start = time.time()
            attack_succ_flag = 0
            Loop=2
            if stop_on_success and steps%3==0:
                if attack_succ == 0:
                    model_tests = self.test(self.workers, self.prompts, include_loss=False)
                    model_tests_jb = model_tests[0]
                    if all(all(int(tests) for tests in model_test) for model_test in model_tests_jb):
                        attack_succ_flag = 1
                    else:
                        attack_succ_flag = 0
                    # print("evaluate one time", time.time()-start)
                else:
                    for check in range(Loop):
                        if model_tests is not None:
                            model_tests = tuple(x[:len(self.targets)] for x in model_tests)
                        else:
                            model_tests = self.test(self.workers, self.prompts, include_loss=False, random_prompt=True)
                        # model_tests = self.test(self.workers, self.prompts, include_loss=False)
                        model_tests_jb = model_tests[0]
                        model_tests = None

                        if all(all(int(tests) for tests in model_test) for model_test in model_tests_jb):
                            attack_succ_flag = 1
                            continue
                        attack_succ_flag = 0
                        break
                
                if attack_succ_flag == 1:
                    # if True:
                    if attack_succ == 1:
                        model_tests = self.test(self.workers, self.prompts, include_loss=True, random_prompt=True) 
                    else:
                        model_tests = self.test(self.workers, self.prompts, include_loss=True)
                    # print('model_tests',model_tests)
                    model_tests_jb = model_tests[0]
                    gen_str = model_tests[3]
                    if all(all(int(tests) for tests in model_test) for model_test in model_tests_jb):
                    # if True:
                            # double check to avoid random
                        self.log(i+1+anneal_from, n_steps+anneal_from, self.control_str, best_loss, runtime, model_tests, gen_str, verbose=verbose)
                        self.log_gen_str(self.prompts[0].inputs_str, gen_str)

                        success = [
                            f"success_cotrol: {self.control_str}\n",
                            f"success_control_toks: {self.control_toks[0].tolist()}\n",
                            f"success_iteration: {i+1+anneal_from}\n",
                            f"success_loss: {best_loss}\n",
                            f"success_generate: {gen_str}\n"
                        ]
                        with open(self.logfile[:-5]+"_succ.txt", "w", encoding='utf-8') as f:
                            f.writelines(success)
                        with open(self.logfile, 'r',) as f:
                            logs = json.load(f)
                        success = {
                            "success_cotrol": self.control_str,
                            "success_iteration": i+1+anneal_from,
                            "success_loss": best_loss,
                            "success_generate": gen_str,
                        }
                        logs["0-success"] = success
                        logs["success_toks"] = {
                            "success_control_toks": self.control_toks[0].tolist(),
                            "success_input_idx": self.prompts[0].eval_toks[0].tolist(),
                        }
                        if attack_succ == 0:
                            success_file = self.logfile[:-5]+"_succ"+self.logfile[-5:]
                            with open(success_file, 'w') as f:
                                json.dump(logs, f, indent=4, sort_keys=True, ensure_ascii=False)
                            attack_succ = 1
                        else:
                            success_file = self.logfile[:-5]+"_random_succ"+self.logfile[-5:]
                            with open(success_file, 'w') as f:
                                json.dump(logs, f, indent=4, sort_keys=True, ensure_ascii=False)
                            attack_random_succ = 1
                            os.remove(self.logfile)
                            break
                        print('attack_succ and attack_random_succ:', attack_succ, attack_random_succ)
            print("evaluate time", time.time()-start)           
        if attack_succ == 0:
            os.rename(self.logfile, self.logfile[:-5]+"_fail"+self.logfile[-5:])
        return self.control_str, loss, steps

    def test(self, workers, prompts, include_loss=False, random_prompt=False):
        for j, worker in enumerate(workers):
            # print('prompts[j]',prompts[j])
            if random_prompt:
                prompts[j].random_prompt()
            worker(prompts[j], "test", worker.model)
        model_tests = [worker.results.get() for worker in workers]
        assert None not in model_tests, "None in model tests"
        model_tests = np.array(model_tests)
        model_tests_jb = model_tests[...,0].tolist()
        model_tests_mb = model_tests[...,1].tolist()
        model_test_gen_str = model_tests[...,2].tolist()
        model_tests_loss = []
        if include_loss:
            for j, worker in enumerate(workers):
                worker(prompts[j], "test_loss", worker.model)
            model_tests_loss = [worker.results.get() for worker in workers]
            assert None not in model_tests_loss, "None in test_loss"

        # each return vale is 2 dims array, the first dim is workers(LLM models), the second line is goals(targets)
        return model_tests_jb, model_tests_mb, model_tests_loss, model_test_gen_str

    def test_all(self):
        all_workers = self.workers + self.test_workers
        all_prompts = [
            self.managers['PM'](
                self.prompt_templates,
                self.instructions,
                self.queries,
                self.context,
                self.targets + self.test_targets,
                self.succ_flags,
                self.fail_flags,
                self.temp,
                worker.tokenizer,
                worker.conv_template,
                self.control_str,
                # self.test_prefixes,
                self.managers
            )
            for worker in all_workers
        ]
        return self.test(all_workers, all_prompts, include_loss=True)
    
    def parse_results(self, results):
        x = len(self.workers)
        i = len(self.context)
        # print(x,i)
        print('results:',results)
        id_id = results[:x, :i].sum()
        id_od = results[:x, i:].sum()
        od_id = results[x:, :i].sum()
        od_od = results[x:, i:].sum()
        return id_id, id_od, od_id, od_od

    def log(self, step_num, n_steps, control, loss, runtime, model_tests, gen_str="", verbose=True):

        prompt_tests_jb, prompt_tests_mb, model_tests_loss = np.array(model_tests[0], dtype=int), np.array(model_tests[1], dtype=int), np.array(model_tests[2])
        # print(prompt_tests_jb.shape, prompt_tests_mb.shape, model_tests_loss.shape)

        # for log files
        # all_goal_strs = self.goals + self.test_goals
        # all_workers = self.workers + self.test_workers
        tests = {
            # all_goal_strs[i]:
            # [
            #     (all_workers[j].model.name_or_path, prompt_tests_jb[j][i], prompt_tests_mb[j][i], model_tests_loss[j][i])
            #     for j in range(len(all_workers))
            # ]
            # for i in range(len(all_goal_strs))
        }
        n_passed = self.parse_results(prompt_tests_jb)
        n_em = self.parse_results(prompt_tests_mb)
        n_loss = self.parse_results(model_tests_loss)
        total_tests = self.parse_results(np.ones(prompt_tests_jb.shape, dtype=int))
        n_loss = [l / t if t > 0 else 0 for l, t in zip(n_loss, total_tests)]

        tests['n_passed'] = n_passed
        tests['n_em'] = n_em
        tests['n_loss'] = n_loss
        tests['total'] = total_tests

        with open(self.logfile, 'r') as f:
            log = json.load(f)

        log['controls'].append(control)
        log['losses'].append(loss)
        log['runtimes'].append(runtime)
        log['tests'].append(tests)

        with open(self.logfile, 'w') as f:
            json.dump(log, f, indent=4, cls=NpEncoder, ensure_ascii=False)

        if verbose:
            output_str = ''
            for i, tag in enumerate(['id_id', 'id_od', 'od_id', 'od_od']):
                if total_tests[i] > 0:
                    output_str += f"({tag}) | Passed {n_passed[i]:>3}/{total_tests[i]:<3} | EM {n_em[i]:>3}/{total_tests[i]:<3} | Loss {n_loss[i]:.4f}\n"
            print((
                f"\n====================================================\n"
                f"Step {step_num:>4}/{n_steps:>4} ({runtime:.4} s)\n"
                f"**ControlSentence** {output_str}\n"
                # f"**ControlSentence** {gen_str}"
                f"control='{control}'\n"
                f"===================================================="
            ))
    def log_gen_str(self, input_str, gen_str):
        print('log_file', self.logfile[:-5])
        for idx, in_str in enumerate(input_str):
            print(f'The {idx}\'s input is: {in_str}')
            for (worker, out_str) in zip(self.workers, gen_str[0]):
                out_str = out_str.replace("\n", "\\n").replace("\t", "\\t")
                print(f'{worker.model_name}\'s output is: {out_str}')
                with open(self.logfile[:-5]+"_gen.txt", "a") as f:
                    f.write(f'{idx}:{worker.model_name}\'s output is: {out_str}\n')
                    f.write(f'control: {self.control_str}\n')  
        print("=========================================================\n\n")
        


class ProgressiveMultiPromptAttack(object):
    """A class used to manage multiple progressive prompt-based attacks."""
    def __init__(self, 
        goals, 
        targets,
        succ_flags,
        fail_flags,
        workers,
        progressive_goals=True,
        progressive_models=True,
        control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        # test_prefixes=["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"],
        logfile=None,
        managers=None,
        test_goals=[],
        test_targets=[],
        test_workers=[],
        *args, **kwargs
    ):

        """
        Initializes the ProgressiveMultiPromptAttack object with the provided parameters.

        Parameters
        ----------
        goals : list of str
            The list of intended goals of the attack
        targets : list of str
            The list of targets of the attack
        workers : list of Worker objects
            The list of workers used in the attack
        progressive_goals : bool, optional
            If true, goals progress over time (default is True)
        progressive_models : bool, optional
            If true, models progress over time (default is True)
        control_init : str, optional
            A string used to control the attack (default is "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
        test_prefixes : list, optional
            A list of prefixes to test the attack (default is ["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"])
        logfile : str, optional
            A file to which logs will be written
        managers : dict, optional
            A dictionary of manager objects, required to create the prompts.
        test_goals : list of str, optional
            The list of test goals of the attack
        test_targets : list of str, optional
            The list of test targets of the attack
        test_workers : list of Worker objects, optional
            The list of test workers used in the attack
        """

        self.goals = goals
        self.targets = targets
        self.succ_flags = succ_flags
        self.fail_flags = fail_flags
        self.workers = workers
        self.test_goals = test_goals
        self.test_targets = test_targets
        self.test_workers = test_workers
        self.progressive_goals = progressive_goals
        self.progressive_models = progressive_models
        self.control = control_init
        # self.test_prefixes = test_prefixes
        self.logfile = logfile
        self.managers = managers
        self.mpa_kwargs = ProgressiveMultiPromptAttack.filter_mpa_kwargs(**kwargs)

        if logfile is not None:
            with open(logfile, 'w') as f:
                json.dump({
                        'params': {
                            'goals': goals,
                            'targets': targets,
                            "succ_flags": succ_flags,
                            "fail_flags": fail_flags,
                            'test_goals': test_goals,
                            'test_targets': test_targets,
                            'progressive_goals': progressive_goals,
                            'progressive_models': progressive_models,
                            'control_init': control_init,
                            # 'test_prefixes': test_prefixes,
                            'models': [
                                {
                                    'model_path': worker.model.name_or_path,
                                    'tokenizer_path': worker.tokenizer.name_or_path,
                                    'conv_template': worker.conv_template.name
                                }
                                for worker in self.workers
                            ],
                            'test_models': [
                                {
                                    'model_path': worker.model.name_or_path,
                                    'tokenizer_path': worker.tokenizer.name_or_path,
                                    'conv_template': worker.conv_template.name
                                }
                                for worker in self.test_workers
                            ]
                        },
                        'controls': [],
                        'losses': [],
                        'runtimes': [],
                        'tests': []
                    }, f, indent=4, ensure_ascii=False
                )

    @staticmethod
    def filter_mpa_kwargs(**kwargs):
        mpa_kwargs = {}
        for key in kwargs.keys():
            if key.startswith('mpa_'):
                mpa_kwargs[key[4:]] = kwargs[key]
        return mpa_kwargs

    def run(self, 
            n_steps: int = 1000, 
            batch_size: int = 1024, 
            topk: int = 256, 
            temp: float = 1.,
            allow_non_ascii: bool = False,
            target_weight = None, 
            control_weight = None,
            anneal: bool = True,
            test_steps: int = 50,
            incr_control: bool = True,
            stop_on_success: bool = True,
            verbose: bool = True,
            filter_cand: bool = True,
        ):
        """
        Executes the progressive multi prompt attack.

        Parameters
        ----------
        n_steps : int, optional
            The number of steps to run the attack (default is 1000)
        batch_size : int, optional
            The size of batches to process at a time (default is 1024)
        topk : int, optional
            The number of top candidates to consider (default is 256)
        temp : float, optional
            The temperature for sampling (default is 1)
        allow_non_ascii : bool, optional
            Whether to allow non-ASCII characters (default is False)
        target_weight
            The weight assigned to the target
        control_weight
            The weight assigned to the control
        anneal : bool, optional
            Whether to anneal the temperature (default is True)
        test_steps : int, optional
            The number of steps between tests (default is 50)
        incr_control : bool, optional
            Whether to increase the control over time (default is True)
        stop_on_success : bool, optional
            Whether to stop the attack upon success (default is True)
        verbose : bool, optional
            Whether to print verbose output (default is True)
        filter_cand : bool, optional
            Whether to filter candidates whose lengths changed after re-tokenization (default is True)
        """


        if self.logfile is not None:
            with open(self.logfile, 'r') as f:
                log = json.load(f)
                
            log['params']['n_steps'] = n_steps
            log['params']['test_steps'] = test_steps
            log['params']['batch_size'] = batch_size
            log['params']['topk'] = topk
            log['params']['temp'] = temp
            log['params']['allow_non_ascii'] = allow_non_ascii
            log['params']['target_weight'] = target_weight
            log['params']['control_weight'] = control_weight
            log['params']['anneal'] = anneal
            log['params']['incr_control'] = incr_control
            log['params']['stop_on_success'] = stop_on_success

            with open(self.logfile, 'w') as f:
                json.dump(log, f, indent=4, ensure_ascii=False)

        num_goals = 1 if self.progressive_goals else len(self.goals)
        num_workers = 1 if self.progressive_models else len(self.workers)
        step = 0
        stop_inner_on_success = self.progressive_goals
        loss = np.infty

        while step < n_steps:
            attack = self.managers['MPA'](
                self.goals[:num_goals], 
                self.targets[:num_goals],
                self.succ_flags[:num_goals],
                self.fail_flags[:num_goals],
                self.workers[:num_workers],
                self.control,
                # self.test_prefixes,
                self.logfile,
                self.managers,
                self.test_goals,
                self.test_targets,
                self.test_workers,
                **self.mpa_kwargs
            )
            if num_goals == len(self.goals) and num_workers == len(self.workers):
                stop_inner_on_success = False
            control, loss, inner_steps = attack.run(
                n_steps=n_steps-step,
                batch_size=batch_size,
                topk=topk,
                temp=temp,
                allow_non_ascii=allow_non_ascii,
                target_weight=target_weight,
                control_weight=control_weight,
                anneal=anneal,
                anneal_from=step,
                prev_loss=loss,
                stop_on_success=stop_inner_on_success,
                test_steps=test_steps,
                filter_cand=filter_cand,
                verbose=verbose
            )
            
            step += inner_steps
            self.control = control

            if num_goals < len(self.goals):
                num_goals += 1
                loss = np.infty
            elif num_goals == len(self.goals):
                if num_workers < len(self.workers):
                    num_workers += 1
                    loss = np.infty
                elif num_workers == len(self.workers) and stop_on_success:
                    model_tests = attack.test_all()
                    attack.log(step, n_steps, self.control, loss, 0., model_tests, verbose=verbose)
                    break
                else:
                    if isinstance(control_weight, (int, float)) and incr_control:
                        if control_weight <= 0.09:
                            control_weight += 0.01
                            loss = np.infty
                            if verbose:
                                print(f"Control weight increased to {control_weight:.5}")
                        else:
                            stop_inner_on_success = False

        return self.control, step

class IndividualPromptAttack(object):
    """ A class used to manage attacks for each target string / behavior."""
    def __init__(self, 
        prompt_templates,
        instructions,
        queries,
        contexts,
        targets,
        train_succ_flags,
        train_fail_flags,
        workers,
        control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        logfile=None,
        managers=None,
        test_goals=[],
        test_targets=[],
        test_workers=[],
        insert_middle = False,
        weighted_update = -1,
        *args,
        **kwargs,
    ):

        """
        Initializes the IndividualPromptAttack object with the provided parameters.

        Parameters
        ----------
        goals : list
            The list of intended goals of the attack
        targets : list
            The list of targets of the attack
        workers : list
            The list of workers used in the attack
        control_init : str, optional
            A string used to control the attack (default is "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
        test_prefixes : list, optional
            A list of prefixes to test the attack (default is ["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"])
        logfile : str, optional
            A file to which logs will be written
        managers : dict, optional
            A dictionary of manager objects, required to create the prompts.
        test_goals : list, optional
            The list of test goals of the attack
        test_targets : list, optional
            The list of test targets of the attack
        test_workers : list, optional
            The list of test workers used in the attack
        """

        self.prompt_templates = prompt_templates
        self.contexts = contexts
        self.instructions = instructions
        self.queries = queries 
        self.targets = targets
        self.train_succ_flags = train_succ_flags
        self.train_fail_flags = train_fail_flags
        self.workers = workers
        self.test_goals = test_goals
        self.test_targets = test_targets
        self.test_workers = test_workers
        self.control = control_init
        self.control_init = control_init
        self.logfile = logfile
        self.managers = managers
        self.insert_middle = insert_middle
        self.weighted_update = weighted_update
        self.mpa_kewargs = IndividualPromptAttack.filter_mpa_kwargs(**kwargs)

        if logfile is not None:
            self.log_dir = logfile
            self.logfile = os.path.join(self.log_dir, "config.json")
            os.makedirs(self.log_dir, exist_ok=True)
            with open(self.logfile, 'w') as f:
                json.dump({
                        'params': {
                            'prompt_templates': prompt_templates,
                            'contexts': contexts,
                            'instructions': instructions,
                            'queries': queries,
                            'targets': targets,
                            'success flag': train_succ_flags,
                            "fail flag": train_fail_flags,
                            'test_goals': test_goals,
                            'test_targets': test_targets,
                            'control_init': control_init,
                            'models': [
                                {
                                    'model_path': worker.model.name_or_path,
                                    'tokenizer_path': worker.tokenizer.name_or_path,
                                    'conv_template': worker.conv_template.name
                                }
                                for worker in self.workers
                            ],
                            'test_models': [
                                {
                                    'model_path': worker.model.name_or_path,
                                    'tokenizer_path': worker.tokenizer.name_or_path,
                                    'conv_template': worker.conv_template.name
                                }
                                for worker in self.test_workers
                            ]
                        },
                        'controls': [],
                        'losses': [],
                        'runtimes': [],
                        'tests': []
                    }, f, indent=4
                )

    @staticmethod
    def filter_mpa_kwargs(**kwargs):
        mpa_kwargs = {}
        for key in kwargs.keys():
            if key.startswith('mpa_'):
                mpa_kwargs[key[4:]] = kwargs[key]
        return mpa_kwargs

    def run(self, 
            n_steps: int = 1000, 
            batch_size: int = 1024, 
            data_offset: int = 0,
            topk: int = 256, 
            temp: float = 1., 
            allow_non_ascii: bool = True,
            target_weight: Optional[Any] = None, 
            control_weight: Optional[Any] = None,
            anneal: bool = True,
            test_steps: int = 50,
            incr_control: bool = True,
            stop_on_success: bool = True,
            verbose: bool = True,
            filter_cand: bool = True
        ):
        """
        Executes the individual prompt attack.

        Parameters
        ----------
        n_steps : int, optional
            The number of steps to run the attack (default is 1000)
        batch_size : int, optional
            The size of batches to process at a time (default is 1024)
        topk : int, optional
            The number of top candidates to consider (default is 256)
        temp : float, optional
            The temperature for sampling (default is 1)
        allow_non_ascii : bool, optional
            Whether to allow non-ASCII characters (default is True)
        target_weight : any, optional
            The weight assigned to the target
        control_weight : any, optional
            The weight assigned to the control
        anneal : bool, optional
            Whether to anneal the temperature (default is True)
        test_steps : int, optional
            The number of steps between tests (default is 50)
        incr_control : bool, optional
            Whether to increase the control over time (default is True)
        stop_on_success : bool, optional
            Whether to stop the attack upon success (default is True)
        verbose : bool, optional
            Whether to print verbose output (default is True)
        filter_cand : bool, optional
            Whether to filter candidates (default is True)
        """

        if self.logfile is not None:
            with open(self.logfile, 'r') as f:
                log = json.load(f)
                
            log['params']['n_steps'] = n_steps
            log['params']['test_steps'] = test_steps
            log['params']['batch_size'] = batch_size
            log['params']['topk'] = topk
            log['params']['temp'] = temp
            log['params']['allow_non_ascii'] = allow_non_ascii
            log['params']['target_weight'] = target_weight
            log['params']['control_weight'] = control_weight
            log['params']['anneal'] = anneal
            log['params']['incr_control'] = incr_control
            log['params']['stop_on_success'] = stop_on_success

            with open(self.logfile, 'w') as f:
                json.dump(log, f, indent=4, ensure_ascii=False)

        stop_inner_on_success = stop_on_success
        
        for i in range(len(self.targets)):
            print(f"Goal {i+1}/{len(self.targets)}")
            log_file = os.path.join(self.log_dir, "attack_"+str(i+data_offset)+".json")
            with open(log_file, 'w') as f:
                res = {
                    'config': {
                        "contexts": self.contexts[i],
                        "queries": self.queries[i],
                        "instructions": self.instructions,
                        "prompt_templates": self.prompt_templates,
                        'targets': self.targets[i],
                        "success_flag": self.train_succ_flags[i],
                        "fail_flag": self.train_fail_flags[i],
                    },
                    'controls' : [],
                    'losses' : [],
                    'runtimes' : [],
                    'tests' : [],
                }
                json.dump(res,f)
            attack = self.managers['MPA'](
                self.prompt_templates,
                self.instructions,
                self.queries[i:i+1], 
                self.contexts[i:i+1],
                self.targets[i:i+1],
                self.train_succ_flags[i],
                self.train_fail_flags[i],
                temp,
                self.workers,
                self.control,
                log_file,
                self.managers,
                self.test_goals,
                self.test_targets,
                self.test_workers,
                self.insert_middle,
                self.weighted_update,
                **self.mpa_kewargs
            )
            attack.run(
                n_steps=n_steps,
                batch_size=batch_size,
                topk=topk,
                temp=temp,
                allow_non_ascii=allow_non_ascii,
                target_weight=target_weight,
                control_weight=control_weight,
                anneal=anneal,
                anneal_from=0,
                prev_loss=np.infty,
                stop_on_success=stop_inner_on_success,
                test_steps=test_steps,
                log_first=True,
                filter_cand=filter_cand,
                verbose=verbose
            )

        return self.control, n_steps

class EvaluateAttack(object):
    """A class used to evaluate an attack using generated json file of results."""
    def __init__(self, 
        goals, 
        targets,
        succ_flags,
        fail_flags,
        workers,
        control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        # test_prefixes=["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"],
        logfile=None,
        managers=None,
        test_goals=[],
        test_targets=[],
        test_workers=[],
        **kwargs,
    ):
        
        """
        Initializes the EvaluateAttack object with the provided parameters.

        Parameters
        ----------
        goals : list
            The list of intended goals of the attack
        targets : list
            The list of targets of the attack
        workers : list
            The list of workers used in the attack
        control_init : str, optional
            A string used to control the attack (default is "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !")
        test_prefixes : list, optional
            A list of prefixes to test the attack (default is ["I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot", "Hello!"])
        logfile : str, optional
            A file to which logs will be written
        managers : dict, optional
            A dictionary of manager objects, required to create the prompts.
        test_goals : list, optional
            The list of test goals of the attack
        test_targets : list, optional
            The list of test targets of the attack
        test_workers : list, optional
            The list of test workers used in the attack
        """

        self.goals = goals
        self.targets = targets
        self.succ_flags = succ_flags
        self.fail_flags = fail_flags
        self.workers = workers
        self.test_goals = test_goals
        self.test_targets = test_targets
        self.test_workers = test_workers
        self.control = control_init
        self.logfile = logfile
        self.managers = managers
        self.mpa_kewargs = IndividualPromptAttack.filter_mpa_kwargs(**kwargs)

        assert len(self.workers) == 1

        if logfile is not None:
            with open(logfile, 'w') as f:
                json.dump({
                        'params': {
                            'goals': goals,
                            'targets': targets,
                            "succ flags": succ_flags,
                            "fail flags": fail_flags,
                            'test_goals': test_goals,
                            'test_targets': test_targets,
                            'control_init': control_init,
                            'models': [
                                {
                                    'model_path': worker.model.name_or_path,
                                    'tokenizer_path': worker.tokenizer.name_or_path,
                                    'conv_template': worker.conv_template.name
                                }
                                for worker in self.workers
                            ],
                            'test_models': [
                                {
                                    'model_path': worker.model.name_or_path,
                                    'tokenizer_path': worker.tokenizer.name_or_path,
                                    'conv_template': worker.conv_template.name
                                }
                                for worker in self.test_workers
                            ]
                        },
                        # 'controls': [],
                        # 'losses': [],
                        # 'runtimes': [],
                        # 'tests': []
                    }, f, indent=4
                )

    @staticmethod
    def filter_mpa_kwargs(**kwargs):
        mpa_kwargs = {}
        for key in kwargs.keys():
            if key.startswith('mpa_'):
                mpa_kwargs[key[4:]] = kwargs[key]
        return mpa_kwargs

    @torch.no_grad()
    def run(self, steps, controls, batch_size):

        model, tokenizer = self.workers[0].model, self.workers[0].tokenizer
        tokenizer.padding_side = 'left'

        if self.logfile is not None:
            with open(self.logfile, 'r') as f:
                log = json.load(f)

            log['params']['num_tests'] = len(controls)

            with open(self.logfile, 'w') as f:
                json.dump(log, f, indent=4, ensure_ascii=False)

        total_jb, total_em, total_outputs = [],[],[]
        test_total_jb, test_total_em, test_total_outputs = [],[],[]
        prev_control = 'haha'
        for step, control in enumerate(controls):
            for (mode, goals, targets) in zip(*[('Train', 'Test'), (self.goals, self.test_goals), (self.targets, self.test_targets)]):
                if control != prev_control:
                    attack = self.managers['MPA'](
                        goals, 
                        targets,
                        self.succ_flags,
                        self.fail_flags,
                        self.workers,
                        control,
                        # self.test_prefixes,
                        self.logfile,
                        self.managers,
                        **self.mpa_kewargs
                    )
                    all_inputs = [p.eval_str for p in attack.prompts[0]._prompts]
                    max_new_tokens = [p.test_new_toks for p in attack.prompts[0]._prompts]
                    targets = [p.target for p in attack.prompts[0]._prompts]
                    all_outputs = []
                    # iterate each batch of inputs
                    for i in range(len(all_inputs) // batch_size + 1):
                        batch = all_inputs[i*batch_size:(i+1)*batch_size]
                        batch_max_new = max_new_tokens[i*batch_size:(i+1)*batch_size]

                        batch_inputs = tokenizer(batch, padding=True, truncation=False, return_tensors='pt')

                        batch_input_ids = batch_inputs['input_ids'].to(model.device)
                        batch_attention_mask = batch_inputs['attention_mask'].to(model.device)
                        # position_ids = batch_attention_mask.long().cumsum(-1) - 1
                        # position_ids.masked_fill_(batch_attention_mask == 0, 1)
                        outputs = model.generate(batch_input_ids, attention_mask=batch_attention_mask, max_new_tokens=max(batch_max_new))
                        batch_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                        all_outputs.extend(batch_outputs)

                        # clear cache
                        del batch_inputs, batch_input_ids, batch_attention_mask, outputs, batch_outputs
                        torch.cuda.empty_cache()
                    
                    curr_jb, curr_em = [], []
                    for (gen_str, target) in zip(all_outputs, targets):
                        # jailbroken = not any([prefix in gen_str for prefix in self.test_prefixes])
                        jailbroken = not any([fail in gen_str for fail in self.fail_flags])
                        jailbroken &= all(succ in gen_str for succ in self.succ_flags)      
                        em = target in gen_str
                        curr_jb.append(jailbroken)
                        curr_em.append(em)
                
                if mode == 'Train':
                    total_jb.append(curr_jb)
                    total_em.append(curr_em)
                    total_outputs.append(all_outputs)
                    # print(all_outputs)
                else:
                    test_total_jb.append(curr_jb)
                    test_total_em.append(curr_em)
                    test_total_outputs.append(all_outputs)

                print(f"{mode} Step {step+1}/{len(controls)} | Jailbroken {sum(curr_jb)}/{len(all_outputs)} | EM {sum(curr_em)}/{len(all_outputs)}")

            prev_control = control

        return total_jb, total_em, test_total_jb, test_total_em, total_outputs, test_total_outputs

def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config



class ModelWorker(threading.Thread):

    def __init__(self, model_path, model_kwargs, tokenizer, conv_template, device, dataType, task_queue, res_queue):
        #edit use AutoPeftModelForCausalLM
        threading.Thread.__init__(self)
        # if "starchat" in model_path:
        #     self.model = pipeline(
        #         "text-generation",
        #         model = model_path,
        #         device_map = "auto",
        #         torch_dtype = dataType,
        #     )
        # else:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dataType,
            trust_remote_code=True,
            **model_kwargs
        )

        # if "quantization_config" not in model_kwargs:
        #     if dataType == torch.float16:
        #         self.model.half()ra
        
        if "device_map" not in model_kwargs:
            self.model.to(device)
        self.model.eval()
            
            
        
        # edit: enable gradient checkpointing and pare for kbit training
        # self.model.gradient_checkpointing_enable()
        # self.model = prepare_model_for_kbit_training(self.model)
        # modules = find_all_linear_names(self.model)
        # peft_config = create_peft_config(modules)
        # self.model = get_peft_model(self.model, peft_config)
        model_path_low = model_path.lower()
        if "codellama" in model_path_low:
            self.model_name="codellama"
        elif "llama" in model_path_low:
            self.model_name="llama"
        elif "vicuna" in model_path_low:
            self.model_name="vicuna"
        elif "mistral" in model_path_low:
            self.model_name = "mistral"
        else:
            self.model_name=model_path_low 
        self.model.name = self.model_name
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.tasks = task_queue
        self.results = res_queue

            
    def run(self):
        print(f"Started worker {self.ident} for model {self.model.name_or_path}")
        try:
            while True:
                task = self.tasks.get()
                # print("model step ???? 0.0 repeat")
                torch.cuda.empty_cache()
                # print("model step ???? 0.1 repeat")
                if task is None:
                    break
                ob, fn, args, kwargs = task
                if fn == "grad":
                    with torch.enable_grad():
                        res = ob.grad(*args, **kwargs)
                        self.results.put(res)
                else:
                    with torch.no_grad():
                        if fn == "logits":
                            self.results.put(ob.logits(*args, **kwargs))
                        elif fn == "contrast_logits":
                            self.results.put(ob.contrast_logits(*args, **kwargs))
                        elif fn == "test":
                            self.results.put(ob.test(*args, **kwargs))
                        elif fn == "test_loss":
                            self.results.put(ob.test_loss(*args, **kwargs))
                        else:
                            self.results.put(fn(*args, **kwargs))
        except Exception as e:
            print(f"Worker {self.ident} failed with exception {e}")
            traceback.print_exc()

            self.results.put(None)
            raise e

    # def start(self):
    #     self.process = mp.Process(
    #         target=ModelWorker.run,
    #         args=(self.tasks, self.results) #todo the model cannot be passed via args, may be a global variable
    #     )
    #     self.process.start()
    #     return self
    
    def stop(self):
        self.tasks.put(None)
        # if self.thread is not None:
        #     self.thread.join()
        self.join()
        torch.cuda.empty_cache()
        return self

    def __call__(self, ob, fn, *args, **kwargs):
        self.tasks.put((deepcopy(ob), fn, args, kwargs))
        return self
    


def get_workers(params, eval=False):
    tokenizers = []
    task_queue = queue.Queue()
    res_queue = queue.Queue()
    print(params.tokenizer_paths)
    for i in range(len(params.tokenizer_paths)):
        print( params.tokenizer_paths[i])
        tokenizer = AutoTokenizer.from_pretrained(
            params.tokenizer_paths[i],
            trust_remote_code=True,
            **params.tokenizer_kwargs[i]
        )
        # tokenizer = AutoTokenizer.from_pretrained("experiments/bigscience")
        if 'oasst-sft-6-llama-30b' in params.tokenizer_paths[i]:
            tokenizer.bos_token_id = 1
            tokenizer.unk_token_id = 0
        if 'guanaco' in params.tokenizer_paths[i]:
            tokenizer.eos_token_id = 2
            tokenizer.unk_token_id = 0
        if 'llama-2' in params.tokenizer_paths[i]:
            # tokenizer.pad_token = tokenizer.unk_token   
            tokenizer.pad_token = "[PAD]"   
            tokenizer.padding_side = 'left'
        if 'falcon' in params.tokenizer_paths[i]:
            tokenizer.padding_side = 'left'
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizers.append(tokenizer)

    print(f"Loaded {len(tokenizers)} tokenizers")

    raw_conv_templates = [
        get_conversation_template(tokenizer_path)
        for template, tokenizer_path in zip(params.conversation_templates, params.tokenizer_paths)
    ]
    conv_templates = []
    for conv in raw_conv_templates:
        if conv.name == 'zero_shot':
            conv.roles = tuple(['### ' + r for r in conv.roles])
            conv.sep = '\n'
        elif conv.name == 'llama-2':
            conv.sep2 = conv.sep2.strip()
        conv_templates.append(conv)
        
    print(f"Loaded {len(conv_templates)} conversation templates")
    workers = [
        ModelWorker(
            params.model_paths[i],
            params.model_kwargs[i],
            tokenizers[i],
            conv_templates[i],
            params.devices[i],
            dataType=params.dataType,
            task_queue=task_queue, res_queue=res_queue
        )
        for i in range(len(params.model_paths))
    ]
    if not eval:
        for worker in workers:
            worker.start()

    num_train_models = getattr(params, 'num_train_models', len(workers))
    print('Loaded {} train models'.format(num_train_models))
    print('Loaded {} test models'.format(len(workers) - num_train_models))

    return workers[:num_train_models], workers[num_train_models:]


def get_goals_and_targets(params):

    queries = getattr(params, 'queries', [])
    contexts = getattr(params, 'contexts', [])
    train_targets = getattr(params, 'targets', [])
    test_goals = getattr(params, 'test_goals', [])
    test_targets = getattr(params, 'test_targets', [])
    offset = getattr(params, 'data_offset', 0)
    if params.train_data:
        with open(params.prompt_template, "r") as f:
            prompt_templates = f.readlines()
        prompt_templates = [line.strip() for line in prompt_templates]
        with open(params.instructions, "r") as f:
            instructions = f.readlines()
        instructions = [line.strip() for line in instructions]

        train_data = pd.read_csv(params.train_data, sep="\^\^")
        # params.n_train_data = len(train_data['target'].tolist())
        queries = train_data['query'].tolist()[offset:offset+params.n_train_data]
        for idx, query in enumerate(queries):
            queries[idx] = query.split("&^&")
            queries[idx] = [q.replace("\\n", "\n").strip() for q in queries[idx]]
        if 'context' in train_data.columns:
            contexts = train_data['context'].tolist()[offset:offset+params.n_train_data]
        else:
            contexts = [""] * len(queries)
        if 'target' in train_data.columns:
            train_targets = train_data['target'].tolist()[offset:offset+params.n_train_data]
        else:
            train_targets = [""] * len(queries)
        for idx, context in enumerate(contexts):
            contexts[idx] = context.replace("\\n", "\n")
            # train_targets[idx] = train_targets[idx].replace("\\n", "\n")

        if 'succ_flag' in train_data.columns:
            train_succ_flags = train_data['succ_flag'].tolist()[offset:offset+params.n_train_data]
            train_fail_flags = train_data['fail_flag'].tolist()[offset:offset+params.n_train_data]
        else:
            train_succ_flags = [""] * len(contexts)
            train_fail_flags = ['sfmixip'] * len(contexts)
        for idx, (succ, fail) in enumerate(zip(train_succ_flags, train_fail_flags)):
            print(succ, fail)
            train_fail_flags[idx] = fail.split("^")
            train_succ_flags[idx] = succ.split("^")
        if params.test_data and params.n_test_data > 0:
            test_data = pd.read_csv(params.test_data)
            test_targets = test_data['target'].tolist()[offset:offset+params.n_test_data]
            if 'goal' in test_data.columns:
                test_goals = test_data['goal'].tolist()[offset:offset+params.n_test_data]
            else:
                test_goals = [""] * len(test_targets)
        elif params.n_test_data > 0:
            test_targets = train_data['target'].tolist()[offset+params.n_train_data:offset+params.n_train_data+params.n_test_data]
            if 'goal' in train_data.columns:
                test_goals = train_data['goal'].tolist()[offset+params.n_train_data:offset+params.n_train_data+params.n_test_data]
            else:
                test_goals = [""] * len(test_targets)

    assert len(contexts) == len(train_targets)
    assert len(test_goals) == len(test_targets)
    print('Loaded {} contexts'.format(len(contexts)))
    print('Loaded {} test goals'.format(len(test_goals)))

    return prompt_templates, instructions, queries, contexts, train_targets, train_succ_flags, train_fail_flags, test_goals, test_targets
