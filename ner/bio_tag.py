tag_def = 'DFNDUM'
tag = {'O': 'O', 'B': 'B-' + tag_def, 'I': 'I-' + tag_def }
tag_ind = {}
for ind, val in enumerate(tag.values()):
    tag_ind[val] = ind + 1

def bio_tagger(title, sent):
    '''
    title: ['word1', 'word2'] a list of words i.e. the title of an wikipedia
    article
    sent: [('word', 'pos')] a sentence in this form thay may or may not contain
    the title.

    Returns:
    A labeled IOB sentence in the format [(word1, pos, iob), ... (wordN, pos, iob)]

    '''
    sent_len = len(sent)
    title_len = len(title)
    # Since there is only one title, it can be defined here just once
    bio_tag = [tag['B']] + [tag['I']]*(title_len - 1)

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
            out_lst.append((*sent[k], tag['O']))
            k += 1
    # finish putting tag_O to the remaining tokens
    out_lst += [(*sent[l], tag['O']) for l in range(k, sent_len )]
    return out_lst

def bio_tkn_tagger(title, sent, int_tags=True):
    '''
    Tags a tokenized (ints) sequence of words
    title: ['tkn1', 'tkn2'] a list of words
    sent: ['tkni'] a sentence in this form
    '''
    sent_len = len(sent)
    title_len = len(title)
    # Since there is only one title, it can be defined here just once
    bio_tag = [tag['B']] + [tag['I']]*(title_len - 1)

    out_lst = []
    k = 0
    while k <= sent_len - title_len :
        matches = all([ sent[k + i] == title[i]
            for i in range(title_len)])
        if matches:
            out_lst += [(sent[k + j], bio_tag[j]) for j in range(title_len)]
            # This advances a total of len(title) tokens
            k += title_len
        else:
            # when there is no match just move on and tag O
            out_lst.append((sent[k], tag['O']))
            k += 1
    # finish putting tag_O to the remaining tokens
    out_lst += [(sent[l], tag['O']) for l in range(k, sent_len )]
    if int_tags:
        out_lst = [(p[0], tag_ind[p[1]]) for p in out_lst]
    return out_lst

