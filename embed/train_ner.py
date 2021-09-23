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


def gen_cfg(**kwargs):
    cfg = {'wiki_src': 'wikipedia/wiki_definitions_improved.xml.gz',
            'pm_src':  'planetmath/datasets/planetmath_definitions.xml.gz',
            'stacks_src': 'stacks-project/datasets/stacks-definitions.xml.gz',
            }
    cfg['base_dir'] = os.environ.get('PERMSTORAGE', '/media/hd1') # This is permanent storage
    cfg['local_dir'] = os.environ.get('TEMPFASTSTORAGE',
            '/tmp/rm_me_NER')  # This is temporary fast storage
    return cfg

def split_fields(elem):
    title = elem.find('.//dfndum').text 
    section = elem.get('name')
    defin = elem.find('.//stmnt').text
    return (title, section, defin)

def get_wiki_pm_stacks_data(cfg):
    wiki = []
    with gzip.open(os.path.join('/media/hd1/', cfg['wiki_src']), 'r') as xml_fobj:
        def_xml = etree.parse(xml_fobj)
        for art in def_xml.findall('definition'):
            data = (art.find('.//dfndum').text, '', art.find('.//stmnt').text)
            wiki.append(data)

    plmath = []
    with gzip.open(os.join.path('/media/hd1/', cfg['pm_src']), 'r') as xml_fobj:
        def_xml = etree.parse(xml_fobj)
        for art in def_xml.findall('article'):
            plmath.append(split_fields(art))
    stacks = []
    with gzip.open(os.path.join('/media/hd1/', cfg['stacks_src']), 'r') as xml_fobj:
        def_xml = etree.parse(xml_fobj)
        for art in def_xml.findall('article'):
            try:
                stacks.append(split_fields(art))
            except AttributeError:
                print('The name of the problematic article is: {}'.format(art.attrib['name']))

    text_lst = wiki + plmath + stacks
    return random.shuffle(text_lst)

def get_sentence_tokzer(cfg):

