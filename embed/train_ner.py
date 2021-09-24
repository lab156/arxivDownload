from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional,\
                      GRU, Dropout,GlobalAveragePooling1D, Conv1D, TimeDistributed,\
                      Input, Concatenate, GlobalMaxPooling1D
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import re
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
import nltk.data
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
from nltk.chunk.util import ChunkScore
import pickle
import math
import string

import sklearn.metrics as metrics
#import matplotlib.pyplot as plt

from nltk.chunk import conlltags2tree, tree2conlltags

import gzip
from lxml import etree
from tqdm import tqdm
import random
from collections import Counter

import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from unwiki import unwiki
import ner
from embed_utils import open_w2v
import clean_and_token_text as clean


def gen_cfg(args=None):
    cfg = {'wiki_src': 'wikipedia/wiki_definitions_improved.xml.gz',
            'pm_src':  'planetmath/datasets/planetmath_definitions.xml.gz',
            'stacks_src': 'stacks-project/datasets/stacks-definitions.xml.gz',
            'punkt_tok_src': 'ner_model/oldie/punkt_params.pickle',
            }
    cfg['base_dir'] = os.environ.get('PERMSTORAGE', '/media/hd1') # This is permanent storage
    cfg['local_dir'] = os.environ.get('TEMPFASTSTORAGE',
            '/tmp/rm_me_NER')  # This is temporary fast storage
    return cfg

def argParse():
    '''
    Parsing all arguments
    '''
    import argparse
    parser = argparse.ArgumentParser(description="Train LSTM model")
    parser.add_argument('--epochs', type=int, default=2,
            help="Number of epochs to train. Overrides default value")
    parser.add_argument('--experiments', type=int, default=2,
            help="Number of experiment loops to do.")
    parser.add_argument('-p', '--profiling', action='store_true',
            help="Set the profiling mode to True (default False)")
    parser.add_argument('-m', '--mini', action='store_true',
            help="Set a small version of the training data set.")
    args = parser.parse_args()
    return args

def split_fields(elem):
    title = elem.find('.//dfndum').text 
    section = elem.get('name')
    defin = elem.find('.//stmnt').text
    return (title, section, defin)

def get_wiki_pm_stacks_data(cfg):
    wiki = []
    with gzip.open(os.path.join(cfg['base_dir'], cfg['wiki_src']), 'r') as xml_fobj:
        def_xml = etree.parse(xml_fobj)
        for art in def_xml.findall('definition'):
            data = (art.find('.//dfndum').text, '', art.find('.//stmnt').text)
            wiki.append(data)

    plmath = []
    with gzip.open(os.join.path(cfg['base_dir'], cfg['pm_src']), 'r') as xml_fobj:
        def_xml = etree.parse(xml_fobj)
        for art in def_xml.findall('article'):
            plmath.append(split_fields(art))
    stacks = []
    with gzip.open(os.path.join(cfg['base_dir'], cfg['stacks_src']), 'r') as xml_fobj:
        def_xml = etree.parse(xml_fobj)
        for art in def_xml.findall(cfg['base_dir']):
            try:
                stacks.append(split_fields(art))
            except AttributeError:
                print('The name of the problematic article is: {}'.format(art.attrib['name']))

    text_lst = wiki + plmath + stacks
    return random.shuffle(text_lst)

def get_sentence_tokzer(cfg):
    '''
    grab a predefined sentence tokenizer
    '''
    with open(os.path.join(cfg['base_dir'], cfg['punkt_tok_src']), 'rb') as punk_fobj:
        trainer_params = pickle.load(punk_fobj)
        sent_tok = PunktSentenceTokenizer(trainer_params)
    logger.info(sent_tok._params.abbrev_types)
    return sent_tok

def gen_sent_tokzer(cfg):
    '''
    Get data and train the Sentence tokenizer
    Uses a standard algorithm (Kiss-Strunk) for unsupervised sentence boundary detection
    '''
    text = ''
    for i in range(3550):
        text += text_lst[i][2]

    trainer = PunktTrainer()
    trainer.INCLUDE_ALL_COLLOCS = True
    trainer.train(text)
    sent_tok = PunktSentenceTokenizer(trainer.get_params())
    return sent_tok


def main():
    logging.basicConfig(filename=os.path.join(cfg['save_path_dir'], 'training.log'),
            level=logging.INFO)
    logger.info("GPU devices: {}".format(list_physical_devices('GPU')))
    logger.info("Length of the xml_lst is: {}".format(len(xml_lst)))

    args = argParse()
    cfg = gen_cfg(args = args)

    text_lst = get_wiki_pm_stacks_data(cfg)
    sent_tok = get_sentence_tokzer(cfg)

    def_lst = ner.bio_tag.put_pos_ner_tags(text_lst, sent_tok)

if __name__ == '__main__':
    main()

