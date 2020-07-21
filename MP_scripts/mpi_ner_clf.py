#!/usr/bin/env python3
# Run with
# mpiexec -n 4 --host 10.0.0.71,10.0.0.72,10.0.0.73,10.0.0.74 python3 /home/pi/arxivDownload/mpi_find_defs.py

from mpi4py import MPI
from random import randint
import time
from itertools import cycle
from lxml import etree
import gzip
import logging
import os
from math import ceil

import multiprocessing as mp
import pickle
from nltk.chunk import ChunkParserI
from ner.chunker import NamedEntityChunker, features
from nltk import pos_tag, word_tokenize
import tarfile

# IMPORT MODULES IN PARENT DIR
import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import parsing_xml as px
from extract import Definiendum
import peep_tar as peep
#from mp_extract import parse_clf_chunk, untar_clf_write

def clf_chunk_write(px_fobj, clf, bio, vzer, tokr):
    pass

comm = MPI.COMM_WORLD
rank = comm.rank
Size = comm.Get_size()
Start = 1991
End = 2020
dir_lst = ['math' + repr(s)[-2:] for s in range(Start, End + 1)]
mnt_path = '/mnt/promath/'

def parse_clf_chunk(file_obj, clf, bio, vzer, tokr):
    '''
    Runs the classifier and chunker on the file_obj
    file_obj: file object

    clf, bio, vzer, tokr: pickled classifiers and tokenizer
    '''
    px = parsing_xml.DefinitionsXML(file_obj)
    ddum = Definiendum(px, clf, bio, vzer, tokr)
    return ddum.root

def untar_clf_write(tarpath, output_dir,  *args):
    '''
    Takes a tarfile and runs parse_class_chunk and writes to output_dir

    tarpath: the path to the tar file with all the articles inside ex.
    1401_001.tar.gz

    output_dir: a path to write down results
    '''
    root = etree.Element('root')
    logging.info('Started working on tarpath=%s'%tarpath)

    for xml_file in peep.tar_iter(tarpath, '.xml'): 
        try:
            tar_fileobj = tar_file.extractfile(xml_file)
            art_tree = parse_clf_chunk(tar_fileobj, *args)
            root.append(art_tree)
        except ValueError as ee:
            logging.debug(' '.join([repr(ee), 'file: ', xml_file, ' is empty']))

    #name of tarpath should be in the format '1401_001.tar.gz'
    gz_filename = os.path.basename(tarpath).split('.')[0] + '.xml.gz' 
    logging.debug('The name of the gz_filename is: %s'%gz_filename)
    gz_out_path = os.path.join(output_dir, gz_filename) 
    with gzip.open(gz_out_path, 'wb') as out_f:
        logging.info("Writing to dfdum zipped file to: %s"%gz_out_path)
        out_f.write(etree.tostring(root, pretty_print=True))


for k,dirname in enumerate(dir_lst):   # dirname: math18
    if k%Size == rank:
        try:
            tar_lst = [p for p in  os.listdir(os.path.join(mnt_path, d)) if '.tar.gz' in p]
        except FileNotFoundError:
            print(' %s Not Found'%d)
            break
        print('I am rpi%s and dealing with dir %s \n'%(rank, d))
        out_path = os.path.join('/tmp/', d)
        try:
            os.mkdir(out_path)
        except FileExistsError as ee:
            print(ee, ' continuing using this directory')
            
        #print(tar_lst)
        #root = etree.Element('root', name=d)
        for tarpath in tar_lst:   # tarpath: 9201_001.tar.gz
            pool = mp.Pool(processes=4)
            out_path = args.output
            arg_lst = [(t, out_path, clf, bio, vzer, tokr) for t in dirname]
            pool.starmap(untar_clf_write, arg_lst)
