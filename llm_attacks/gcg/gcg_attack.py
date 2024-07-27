import gc
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from llm_attacks import AttackPrompt, MultiPromptAttack, PromptManager
from llm_attacks import get_embedding_matrix, get_embeddings




class GCGAttackPrompt(AttackPrompt):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
    def grad(self, model):
        # print("!!! gradients evalute with:\n", self.tokenizer.decode(self.input_ids))
        # print("="*35)
        if self.weighted_update != -1:
            res = self.token_gradients(
                model, 
                self.input_ids.to(model.device), 
                input_slice = self._control_slice, 
                target_slice = self._target_slice, 
                loss_slice = self._loss_slice,
                weights = self.weights.to(model.device),
            )
        else:
            res = self.token_gradients(
                model, 
                self.input_ids.to(model.device), 
                input_slice = self._control_slice, 
                target_slice = self._target_slice, 
                loss_slice = self._loss_slice,
            )
        return res

    def token_gradients(self, model, input_ids, input_slice, target_slice, loss_slice, weights=None):

        """
        Computes gradients of the loss with respect to the coordinates.
        
        Parameters
        ----------
        model : Transformer Model
            The transformer model to be used.
        input_ids : torch.Tensor
            The input sequence in the form of token ids.
        input_slice : slice
            The slice of the input sequence for which gradients need to be computed.
        target_slice : slice
            The slice of the input sequence to be used as targets.
        loss_slice : slice
            The slice of the logits to be used for computing the loss.

        Returns
        -------
        torch.Tensor
            The gradients of each token in the input_slice with respect to the loss.
        """

        embed_weights = get_embedding_matrix(model) # 32000, 4096 for llama2
        one_hot = torch.zeros(
            input_ids[input_slice].shape[0],
            embed_weights.shape[0],
            device=model.device,
            dtype=embed_weights.dtype
        )
        one_hot.scatter_(
            1, 
            input_ids[input_slice].unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
        )
        one_hot.requires_grad_()   # build input one hot, so we can use it to calculate gradients
        input_embeds = (one_hot @ embed_weights).unsqueeze(0) # embed input to a (20, 4096) dim vector
        
        # now stitch it together with the rest of the embeddings
        embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach() 
        full_embeds = torch.cat(
            [
                embeds[:,:input_slice.start,:], 
                input_embeds, 
                embeds[:,input_slice.stop:,:]
            ], 
            dim=1) # insert control into middle of input
        
        logits = model(inputs_embeds=full_embeds).logits
        targets = input_ids[target_slice]

        loss_func = nn.CrossEntropyLoss()
        if weights is not None:
            max_weights = weights.max()
            min_weights = weights.min() 
            logits_x = logits[0,loss_slice,:]
            logits_x = logits_x/self.temp
            loss = loss_func(logits_x[weights==max_weights], targets[weights==max_weights]) * max_weights
            loss += loss_func(logits_x[weights==min_weights], targets[weights==min_weights]) * min_weights
        else:
            logits_x = logits
            logits_x = logits_x/self.temp
            loss = loss_func(logits[0,loss_slice,:], targets)
        loss.backward()

        res = one_hot.grad.clone()
        del one_hot, loss, targets, logits, full_embeds, embeds, input_embeds, loss_func, logits_x
        return res
    

class GCGPromptManager(PromptManager):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def sample_control(self, grad, batch_size, topk=256, temp=1, allow_non_ascii=True):  
        # build tensor of [batch, len(control_toks)] with random selected tokens from topk tokens.
        # In each line of retrun value, we mutate the tocken at one position in control sentence, replacing it with a random seleced value in topk tokens at that position. 
        # from line 0 to batch_size/len(control), we mutate the first word, in batch_size/len(control)+1 to batch_size/len(control)*2, we replace the second work, and so on. 

        if not allow_non_ascii: 
            grad[:, self._nonascii_toks.to(grad.device)] = np.infty  # grad shape [20,32000] 
        top_indices = (-grad).topk(topk, dim=1).indices # top_indices shape [20,256] 
        control_toks = self.control_toks.to(grad.device)  # control_toks shape [20:1] 
        original_control_toks = control_toks.repeat(batch_size, 1) # original_control_toks repeat control_toks batch_size times, shape (20) -> (512,20) 
        new_token_pos = torch.arange(
            0, 
            len(control_toks), 
            len(control_toks) / batch_size,  # why use batch size
            device=grad.device
        ).type(torch.int64) # new_token_pos shape [batch,1], looks like [0,0, ..., 1,1, ..., 2,2, ..., control_len, control_len] 
        new_token_val = torch.gather(
            top_indices[new_token_pos], 1,  # top_indices[new_token_pos] shape [512(batch), 256(topk)]
            torch.randint(0, topk, (batch_size, 1),
            device=grad.device)  # random size (512, 1), number from 0 to topk 
        ) # new_token_val shape [batch,1]

        # after gather, we get a tensor of [batch,1], random select batch_size / len(control_toks) of elements from each line in top_indices.  and concat to a tensor of [batch,1]. (batch_size / len(control_toks) = 25.6)
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        return new_control_toks # (512, 20)
        


class GCGMultiPromptAttack(MultiPromptAttack):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def step(self, 
             batch_size=1024, 
             topk=256, 
             temp=1, 
             allow_non_ascii=True, 
             target_weight=1, 
             control_weight=0.1, 
             verbose=False, 
             opt_only=False,
             filter_cand=True,
             is_random=False):

        
        # GCG currently does not support optimization_only mode, 
        # so opt_only does not change the inner loop.
        opt_only = False
        main_device = self.models[0].device
        control_cands = []
        start = time.time()
        if is_random:
            Loop = 3
        else:
            Loop = 1
        grad = None
        
        while(Loop>0):
            if is_random:
                for j, worker in enumerate(self.workers):
                    self.prompts[j].random_prompt()
                    worker(self.prompts[j], "grad", worker.model)
            else:
                for j, worker in enumerate(self.workers):
                    worker(self.prompts[j], "grad", worker.model)

            # Aggregate gradients
           
            for j, worker in enumerate(self.workers):
                
                new_grad = worker.results.get()
                assert new_grad is not None, "None in grad results"
                new_grad = new_grad.to(main_device)
                new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
                if grad is None:
                    grad = torch.zeros_like(new_grad)
                if grad.shape != new_grad.shape:
                    with torch.no_grad():
                        control_cand = self.prompts[j-1].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
                        # print('control_cand:', control_cand)
                        control_cands.append(self.get_filtered_cands(j-1, control_cand, filter_cand=filter_cand, curr_control=self.control_str, succ_sentence=self.succ_flags[j-1]))
                    grad = new_grad
                else:
                    grad += new_grad
            Loop-=1
        # print("grad time:", time.time()-start)
        start = time.time()
        with torch.no_grad():
            control_cand = self.prompts[j].sample_control(grad, batch_size, topk, temp, allow_non_ascii)  ## mutate batch_size of control sentences, each control sentence has len(control_toks) tokens, and each token is replaced by a random selected token from topk tokens.
       
            control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=self.control_str, succ_sentence=self.succ_flags[j]))

        del grad, control_cand ; gc.collect()
        

        # Search
        loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
        with torch.no_grad():
            for j, cand in enumerate(control_cands):
                # Looping through the prompts at this level is less elegant, but
                # we can manage VRAM better this way
                progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else enumerate(self.prompts[0])
                for i in progress:
                    # print('prompts:',self.prompts)
                    for k, worker in enumerate(self.workers):
                        worker(self.prompts[k][i], "logits", worker.model, cand, return_ids=True)
                    res = [worker.results.get() for worker in self.workers]
                    assert None not in res, "None in logits results"
                    logits, ids = zip(*res)
                    loss[j*batch_size:(j+1)*batch_size] += sum([
                        target_weight*self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device) 
                        for k, (logit, id) in enumerate(zip(logits, ids))
                    ])
                    if control_weight != 0:
                        loss[j*batch_size:(j+1)*batch_size] += sum([
                            control_weight*self.prompts[k][i].control_loss(logit, id).mean(dim=-1).to(main_device)
                            for k, (logit, id) in enumerate(zip(logits, ids))
                        ])
                    del logits, ids ; gc.collect()
                    
                    if verbose:
                        progress.set_description(f"loss={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")
            min_idx = loss.argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size

            next_control, cand_loss = control_cands[model_idx][batch_idx], loss[min_idx]
            
        del control_cands, loss ; gc.collect()
        print('Current length:', len(self.workers[0].tokenizer(next_control).input_ids[1:]))
        print(next_control)


        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers)
