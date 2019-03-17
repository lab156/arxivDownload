tag_def = 'DFNDUM'
tag_O = 'O'
tag_B = 'B-' + tag_def
tag_I = 'I-' + tag_def

def bio_tagger(title, sent):
    '''
    title: ['word1', 'word2'] a list of words
    sent: [('word', 'pos')] a sentence in this form
    '''
    sent_len = len(sent)
    title_len = len(title)
    # Since there is only one title, it can be defined here just once
    bio_tag = [tag_B] + [tag_I]*(title_len - 1)

    out_lst = []
    k = 0
    while k <= sent_len - title_len :
        matches = all([ sent[k + i][0].lower() == title[i].lower()
            for i in range(title_len)])
        if matches:
            out_lst += [(*sent[k + j], bio_tag[j]) for j in range(title_len)]
            # This advances a total of len(title) tokens
            k += title_len
        else:
            # when there is no match just move on and tag O
            out_lst.append((*sent[k], tag_O))
            k += 1
    # finish putting tag_O to the remaining tokens
    out_lst += [(*sent[l], tag_O) for l in range(k, sent_len )]
    return out_lst

