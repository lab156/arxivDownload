#!/usr/bin/env python3
from mpi4py import MPI
#import multiprocessing as mp
import datetime as dt
import functools
from glob import glob
import toml
config = toml.load('../config.toml')

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, os.path.join(parentdir, 'embed'))
sys.path.insert(0, parentdir)

import embed.clean_and_token_text as ctt

def slow_fun(mach, jobnum):
    """
    Takes about 4 sec on rpi
    """
    N = dt.datetime.now()
    #proc = mp.current_process()
    proc = 0
    print("At {}, jobnum = {} machine {} started on proc {}"\
            .format((N.hour,N.minute,N.second), jobnum, mach, proc))
    sum([i*i for i in range(10_000_000)])
    return 0


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.Get_size()


    for i in range(5):
        if i%size == rank:
            slow_fun(rank, i)

def main_xml2xml():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.Get_size()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('in_files', type=str, nargs='+',
            help='one or more tar.gz files of Latexmled file (from promath)')
    parser.add_argument('out_dir', type=str,
            help='replicate the in_files dir struct in this directory with the clean files')
    #parser.add_argument('--phrases_file', default=None, type=str, nargs='+',
    #        help='XML file with the phrases to be joined (from the glossary dir)')
    parser.add_argument('--norm_args', nargs='?', default=['rm_punct'],
            help='arguments for the tokenization function')
    parser.add_argument('--num_phrases', type=int, default=0,
            help='Max number of phrases to use')
    #parser.add_argument('--skip_n', default=1, type=int)
    args = parser.parse_args()

    #phrase_lst = ReadGlossary(args.phrases_file).common_phrases_lst(args.num_phrases)
    #join_fun = lambda s: token_phrases3(s, phrase_lst=phrase_lst)

    if rank == 0:
        RG = ctt.ReadGlossary(os.path.join(config['paths']['data'], 'glossary/v3/math*/*.xml.gz'),
                os.path.join(config['paths']['data'], 'glossary/NN.v1/math*/*.xml.gz'))
        ph_dict = RG.first_word_dict(intersect = 'relative', max_phrases=args.num_phrases)
        print(f'Using {len(ph_dict)} phrases')
    else:
        ph_dict = None
    ph_dict = comm.bcast(ph_dict, root=0)
    join_fun = functools.partial(ctt.join_phrases, phrase_dict=ph_dict)
        


    #for j,gz_file in enumerate(args.in_files):
    for j,gz_file in enumerate(glob(args.in_files[0])):
        if j%size == rank:
            N = dt.datetime.now()
            print("At {}, jobnum = {} machine {} started"\
                    .format((N.hour,N.minute,N.second), j, rank))
            ctt.join_xml_para_and_write(gz_file, args.out_dir, join_fun)

    #with mp.Pool(processes=2, maxtasksperchild=1, initializer=worker_init, initargs=(join_fun,)) as pool:
    #    pool.starmap(join_xml_para_and_write, [(f, args.out_dir, _func) for f in args.in_files])

if __name__ == "__main__":
    main_xml2xml()
