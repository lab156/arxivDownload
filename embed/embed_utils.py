import argparse
import numpy as np
import sys
from math import floor
from collections import OrderedDict as odict
from contextlib import contextmanager
import struct as st
import functools
import re

@contextmanager
def open_w2v(filename):
    mfobj = open(filename, 'rb') 
    try:
        m = mfobj.read()
        #print(m[0].decode('utf8'))
        #s = st.Struct('ii')
        #m_it = m.__iter__()
        head_dims = st.unpack('<11s', m[:11])
        n_vocab, n_dim = map(int,head_dims[0].strip().split())
        print(f"Vocabulary size: {n_vocab} and dimension of embed: {n_dim}")
        embed = {}
        #[next(m_it) for _ in range(11)]
        cnt = 11
        for line_cnt in range(n_vocab):
            word = ''
            while True:
                next_char = st.unpack('<1s', m[cnt:cnt+1])[0].decode('utf8')
                cnt += 1
                if next_char == ' ':
                    break
                else:
                    word += next_char
            #print(word)
            vec = np.zeros(n_dim)
            for k in range(n_dim):
                vec[k] = st.unpack('<f', m[cnt:cnt+4])[0]
                cnt += 4
            assert st.unpack('<1s', m[cnt:cnt+1])[0] == b'\n'
            cnt +=1
            embed[word] = vec
        yield embed
    finally:
        mfobj.close()


def generate(vect_dict):
    '''
    get a sample of approximately N words out of the vector file
    '''

    # don't need the vocabulary
    #with open(args.vocab_file, 'r') as f:
    #    words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    lengths = {}
    for v in vect_dict:
        sumv = np.sum(np.abs(vect_dict[v]))
        probs = vect_dict[v]/sumv

        lengths[v] = np.sqrt(np.sum([np.float(x)**2 for x in probs]))
            
    return odict(sorted(lengths.items(), key=lambda t: t[1]))

def nearest(word_vec, unit_embed, n_near=10):
    '''
    Returns the `n_near` closest vectors to the `word_vec` vector
    NOTE that `unit_embed` needs to be unitary vectors
    '''
    dist_dict = {}
    unit_word_vec = word_vec/np.linalg.norm(word_vec)
    for w, v in unit_embed.items():
        #dist_dict[w] = cos_dist(v, word_vec)
        dist_dict[w] = unit_word_vec.dot(v)
    return sorted(dist_dict.items(), key=lambda pair: pair[1], reverse=True)[:n_near]

def normalize_text(text):
    '''
    a copy of the normalize_text function in the word2vec repo
    see the `demo-train-big-model.sh` file
    run tests with python3 -m doctest -v embed_utils.py

    >>> normalize_text('hi, there.')
    'hi , there . '

    >>> normalize_text('This |is work=ing')
    'this  is work ing'

    >>> normalize_text('remove the <br> <br /> <br     />')
    'remove the      '

    >>> normalize_text('en 1823, Colon llego a ?')
    'en   , colon llego a  ? '

    >>> normalize_text('I rem/ember káhler painlevé in § 74')
    'i rem / ember khler painlev in   '
    '''

    repl_list = [text,
            ("’","'") ,
            ("′","'") ,
           ("''", " "),
            ("'"," ' ") ,
            ("“",'"') ,
            ('"',' " ') ,
            ('.',' . ') ,
            (', ',' , ') ,
            ('(',' ( ') ,
            (')',' ) ') ,
            ('!',' ! '),
            ('?',' ? ') ,
            (';',' ') ,
            (':',' ') ,
            ('-',' - ') ,
            ('=',' ') ,
            ('=',' ') ,
            ('*',' ') , 
            ('|',' ') ,
            ('/',' / ') ,
            ('«',' ') ,
            ('»', ' ')]
    text = functools.reduce(lambda a,b: a.replace(*b), repl_list)

    text = re.sub(r'<br\s*/? ?>', ' ', text) # remove <br /> variants
    text = re.sub(r'[0-9]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    return text.lower()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('words', nargs='+')
    parser.add_argument('--out_file', default='vocab.txt', type=str,
            help='file to write the sorted lengths of the vectors')
    parser.add_argument('--vectors_file', default='vectors.txt', type=str)
    #parser.add_argument('--skip_n', default=1, type=int)
    args = parser.parse_args()
    with open(args.vectors_file, 'r') as f:
        vec_dict = {}
        for index, line in enumerate(f):
            vals = line.rstrip().split(' ')
            if vals[0] in args.words:
                vec_dict[vals[0]] = np.array([np.float(k) for k in vals[1:]])

    sorted_dict = generate(vec_dict)
    with open(args.out_file, 'a') as out_f:
        for o in sorted_dict:
            out_f.write("{:<15} {}\n".format(o, sorted_dict[o]))




