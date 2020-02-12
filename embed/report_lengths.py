import argparse
import numpy as np
import sys
from math import floor
from collections import OrderedDict as odict

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

