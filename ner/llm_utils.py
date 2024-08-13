def get_words_back(tok_lst, preds=None, init_str = '##', special_tokens=None):
    '''
    Joins the tokens into words
    tok_lst is in the format ['Hi', 'my', 'Na', '##me']
    word_lst is ['Hi', 'my', 'Name']
    
    for special_tokens, use special_tokens=tokenizer.special_tokens_map.values()
    
    words are labeled by the label of the first token.
    '''
    if preds == None:
        preds = ['O' for _ in tok_lst]
    
    word_lst = []
    iob_lst = []
    idx = 0
    while ( idx < len(tok_lst) ):
        aggregate_str = tok_lst[idx]
        iob_lst.append(preds[idx])
        idx += 1
        while (
            idx < len(tok_lst) and
            tok_lst[idx].startswith(init_str)
               ):
            aggregate_str += tok_lst[idx][2:]
            idx += 1
        word_lst.append(aggregate_str)
    if special_tokens is not None:
        iob_lst = [iob_lst[k] for k,w in enumerate(word_lst) if w
                   not in special_tokens]
        word_lst = [w for w in word_lst if w
                   not in special_tokens]
    return word_lst, iob_lst

def join_math_tokens(word_lst, preds=None):
    '''
    joins the subsequence of tokens '_', 'inline', '_', 'math', '_'
                               and  '_', 'display', '_', 'math', '_'
    '''
    if preds == None:
        preds = ['O' for _ in word_lst]
        
    ret_words = []
    ret_preds = []
    j = 0
    while j < len(word_lst):
        ret_words.append(word_lst[j])
        ret_preds.append(preds[j])
        if word_lst[j] == '_' and len(word_lst) - j > 4:
            if word_lst[j + 1] == 'inline':
                if (
                    word_lst[j + 2] == '_' and
                    word_lst[j + 3] == 'math' and
                    word_lst[j + 4] == '_' 
                ):
                    ret_words[-1] = '_inline_math_'
                    j += 4
            elif word_lst[j + 1] == 'display':
                if (
                    word_lst[j + 2] == '_' and
                    word_lst[j + 3] == 'math' and
                    word_lst[j + 4] == '_' 
                ):
                    ret_words[-1] = '_display_math_'
                    j += 4
        j += 1
    return ret_words, ret_preds

def join_by_example(tokens_, golds_, preds=None):
    if preds == None:
        preds = ['O' for _ in tokens_]
    else:
        preds = preds.copy()
    tokens = tokens_.copy()
    golds = golds_.copy()
    assert len(tokens) == len(preds), "Tokens and preds have different lengths"
    ret_lst = []
    ret_pred_lst = []
    #if len(tokens) > len(golds):
    tokens.reverse()
    preds.reverse()
    j = 0
    while j < min(len(golds), len(tokens)):
        try:
            tok = tokens.pop()
            pred = preds.pop()
        except IndexError:
            import pdb
            pdb.set_trace()

        join_str = tok
        temp_pred_lst = [pred,]
        while (golds[j].startswith(join_str) and 
               join_str != golds[j] and 
               j < len(golds)):
            #j += 1
            try:
                join_str += tokens.pop()
            except IndexError:
                print(f"""Truncation event {len(golds_)=} 
                          {len(ret_lst)=}""")
                j = len(golds)+1
                break
            temp_pred_lst.append(preds.pop())
        ret_lst.append(join_str)
        # Determine the best representation of the merged tokens
        if 'B-DFNDUM' in temp_pred_lst:
            temp_pred_str = 'B-DFNDUM'
        elif 'I-DFNDUM' in temp_pred_lst:
            temp_pred_str = 'I-DFNDUM'
        else:
            temp_pred_str = 'O'
        ret_pred_lst.append(temp_pred_str)
        j += 1
        #ret_lst.append(tokens.pop())
    #else:
    #    ret_lst = tokens
    #    ret_pred_lst = preds
    #    assert len(tokens) == len(golds), \
    #            f"""length of predictions is less than golds {tokens=} \n
    #                      {golds=}"""
    return ret_lst, ret_pred_lst

def get_entity(words, preds, 
        labels=['O', 'B-DFNDUM', 'I-DFNDUM']):
    '''
    returns a list of terms (definienda) from a labeled sentence
    expect a list that has been through get_words_back and join_math_tokens first
    '''
    assert len(words) == len(preds), "Length of words and preds should be the same"

    term_lst = []
    k = 0 # main index
    while k < len(preds):
        cnt = 1
        if preds[k] == labels[1] or labels[2]:
            term = words[k]
            while (k + cnt < len(preds) and 
                    preds[k + cnt] == labels[2]):
                term += ' ' + words[k + cnt]
                cnt += 1
            term_lst.append(term)
        k = k + cnt
    return term_lst

def crop_terms(tokens, preds, **kwargs):
    words, preds = get_words_back(tokens, preds=preds, 
                      special_tokens=kwargs['special_tokens'])
    golds = kwargs.get('golds', None)

    if golds is None:
        words, preds = join_math_tokens(words, preds=preds)
    else:
        words, preds = join_by_example(words, golds, preds=preds)

    term_lst = get_entity(words, preds)
    return list(set(term_lst))


