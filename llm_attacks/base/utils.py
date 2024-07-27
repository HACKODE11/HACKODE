def find_token(tokenizer, sentence_ids, subsentence):        
    subsentence = tokenizer.decode(tokenizer.encode(subsentence))
    subsentence = subsentence.replace("<s> ", "")
    subsentence = subsentence.replace("<s>", "")
    
    results = []
    current = 0
    while True:
        diff_start = -1
        diff_end = -1
        flag = False
        for idx, tok in enumerate(sentence_ids[current:]): 
            subword = tokenizer.decode(tok)
           
            if subword.strip() == '':
                continue
            if subsentence.startswith(subword.strip()):
                if tokenizer.decode(sentence_ids[idx+current:]).strip().startswith(subsentence):
                    
                    flag = True
                    diff_start = idx+current
                    break
       
        if flag is False or diff_start == -1:
            break
        
        for idx, tok in enumerate(sentence_ids[diff_start:]):
            cur_sentence = tokenizer.decode(sentence_ids[diff_start:diff_start+idx+1])
            cur_sentence_shorter = tokenizer.decode(sentence_ids[diff_start:diff_start+idx])
            
            if len(cur_sentence) >= len(subsentence) and len(cur_sentence_shorter) < len(subsentence):
                diff_end = diff_start + idx + 1
                current = diff_end + 1
                break
        
        results.append((diff_start, diff_end))
        # print('results:',results)
        assert flag is True and diff_end != -1, "why flag is True but the end is not found?"
        

    return results
    
