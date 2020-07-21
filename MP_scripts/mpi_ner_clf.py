#!/usr/bin/env python3
# Run with
# mpiexec -np 2 --bind-to none --host 10.0.0.71,10.0.0.72 python3 /home/pi/arxivDownload/MP_scripts/mpi_ner_clf.py

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
from ner.chunker import NamedEntityChunker, features
#from mp_extract import parse_clf_chunk, untar_clf_write

comm = MPI.COMM_WORLD
rank = comm.rank
Size = comm.Get_size()
Start = 1991
End = 2020
dir_lst = ['math' + repr(s)[-2:] for s in range(Start, End + 1)]
cfg = {'mnt_path': '/mnt/promath/',
        'save_path':'/home/pi/glossary',
        'clf': '/mnt/PickleJar/sgd_clf_21-07.pickle',
        'bio': '/mnt/PickleJar/chunker.pickle', 
        'vzer': '/mnt/PickleJar/hash_vect_21-07.pickle',
        'tokr': '/mnt/PickleJar/tokenizer.pickle'}

logging.basicConfig(level = logging.WARNING)

#TODO:

def parse_clf_chunk(file_obj, clf, bio, vzer, tokr, max_stale_tries=15):
    '''
    Runs the classifier and chunker on the file_obj
    file_obj: file object

    clf, bio, vzer, tokr: pickled classifiers and tokenizer

    max_stale_tries: number of retries of OSError Stale file handle
    '''
    retried = 0
    while retried < max_stale_tries:
        retried += 1
        try:
            DD = px.DefinitionsXML(file_obj)
            ddum = Definiendum(DD, clf, bio, vzer, tokr)
            break
        except OSError as ee:
            wait_delay = randint(5,15)
            logging.warning(f"{ee} waiting for {wait_delay} retry: {retried}")
            time.sleep(wait_delay)
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

    for fname, tar_fileobj in peep.tar_iter(tarpath, '.xml'): 
        try:
            art_tree = parse_clf_chunk(tar_fileobj, *args)
            root.append(art_tree)
        except ValueError as ee:
            logging.debug(' '.join([repr(ee), 'file: ', fname, ' is empty']))

    #name of tarpath should be in the format '1401_001.tar.gz'
    gz_filename = os.path.basename(tarpath).split('.')[0] + '.xml.gz' 
    logging.debug('The name of the gz_filename is: %s'%gz_filename)
    gz_out_path = os.path.join(output_dir, gz_filename) 
    with gzip.open(gz_out_path, 'wb') as out_f:
        logging.info("Writing to dfdum zipped file to: %s"%gz_out_path)
        out_f.write(etree.tostring(root, pretty_print=True))

with open(cfg['clf'], 'rb') as class_f:
    clf = pickle.load(class_f)
with open(cfg['bio'], 'rb') as class_f:
    bio = pickle.load(class_f)
with open(cfg['vzer'], 'rb') as class_f:
    vzer = pickle.load(class_f)
with open(cfg['tokr'], 'rb') as class_f:
    tokr = pickle.load(class_f)

for k,dirname in enumerate(dir_lst):   # dirname: math18
    if k%Size == rank:
        print('I am rpi%s and dealing with dir %s \n'%(rank, dirname))
        if k<Size: time.sleep(rank*20) # Only wait the first time around
        try:
            full_path = os.path.join(cfg['mnt_path'], dirname)
            tar_lst = [os.path.join(full_path, p) for p in  os.listdir(full_path) if '.tar.gz' in p]
        except FileNotFoundError:
            print(' %s Not Found'%d)
            break
        out_path = os.path.join(cfg['save_path'], dirname)
        os.makedirs(out_path, exist_ok=True)
            
        #print(tar_lst)
        #root = etree.Element('root', name=d)
        with mp.Pool(processes=3, maxtasksperchild=1) as pool:
            arg_lst = [(t, out_path, clf, bio, vzer, tokr) for t in tar_lst]
            logging.debug(f'First elements of arg_lst are: {arg_lst[:5]}')
            pool.starmap(untar_clf_write, arg_lst)
