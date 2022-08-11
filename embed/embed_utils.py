import argparse
import numpy as np
import sys
from math import floor
from collections import OrderedDict as odict
from contextlib import contextmanager
import struct as st
import functools
import re
from tqdm import tqdm

@contextmanager
def open_w2v(filename):
    mfobj = open(filename, 'rb') 
    
    try:
        m = mfobj.read()
        # get length of first line
        length1st = 0
        while m[length1st] != ord('\n') and length1st < 100:
            length1st += 1
        if length1st == 100:
            raise ValueError('First line lenght could no be found')
        
        #print(m[0].decode('utf8'))
        #s = st.Struct('ii')
        #m_it = m.__iter__()
        head_dims = st.unpack(f'<{length1st}s', m[:length1st])
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

def open_glove(filepath):
    '''
    Input: the path to the *directory* of a GloVe trained data

    Output: embed dictionary with the format: {word: vector}
    '''
    glove_dir_path = filepath
    with open(glove_dir_path + 'vocab.txt', 'r') as f: 
        words = [x.rstrip().split(' ')[0] for x in f.readlines()] 
    with open(glove_dir_path + 'vectors.txt', 'r') as f:
        #vectors = {}
        embed = {}
        for k,line in tqdm(enumerate(f)):
            vals = line.rstrip().split(' ')
            #vectors[vals[0]] = [float(x) for x in vals[1:]]
            try:
                embed[words[k]] = np.array([float(x) for x in vals[1:]])
            except IndexError:
                 print('<unk> was referenced and defined')
                 embed['<unk>'] = np.array([float(x) for x in vals[1:]]) 
    return embed



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
    if isinstance(word_vec, str):
        word_vec = unit_embed[word_vec]
    unit_word_vec = word_vec/np.linalg.norm(word_vec)
    for w, v in unit_embed.items():
        #dist_dict[w] = cos_dist(v, word_vec)
        dist_dict[w] = unit_word_vec.dot(v)
    return sorted(dist_dict.items(), key=lambda pair: pair[1], reverse=True)[:n_near]


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


