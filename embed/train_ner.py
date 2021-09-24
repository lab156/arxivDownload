from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional,\
                      GRU, Dropout,GlobalAveragePooling1D, Conv1D, TimeDistributed,\
                      Input, Concatenate, GlobalMaxPooling1D
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.config import list_physical_devices
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
from datetime import datetime as dt

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

import logging
logger = logging.getLogger(__name__)

def gen_cfg(args=None):
    cfg = {'wiki_src': 'wikipedia/wiki_definitions_improved.xml.gz',
            'pm_src':  'planetmath/datasets/planetmath_definitions.xml.gz',
            'stacks_src': 'stacks-project/datasets/stacks-definitions.xml.gz',
            'punkt_tok_src': 'trained_models/ner_model/oldie/punkt_params.pickle',
            'glob_data_src': '/train_ner/math*/*.xml.gz',
            'embed_data_src':  'embeddings/model4ner_19-33_02-01/vectors.bin',
            }
    cfg['base_dir'] = os.environ.get('PERMSTORAGE', '/media/hd1') # This is permanent storage
    cfg['local_dir'] = os.environ.get('TEMPFASTSTORAGE',
            '/tmp/rm_me_NER')  # This is temporary fast storage

    hoy = dt.now()
    timestamp = hoy.strftime("%b-%d_%H-%M")
    cfg['save_path_dir'] = os.path.join(cfg['local_dir'],
            'trained_ner/lstm_ner/ner_' + timestamp)
    os.makedirs(cfg['save_path_dir'], exist_ok=True)

    cfg['padseq'] = {'maxlen': 50 , 'padding': 'post', 'truncating': 'post'}
    cfg['n_tags'] = 2
    cfg['train_test_split'] = 0.9
    cfg['nbin_feats'] = 2 # number of binary features defined in the function above

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
    with gzip.open(os.path.join(cfg['base_dir'], cfg['pm_src']), 'r') as xml_fobj:
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
    random.shuffle(text_lst)
    return text_lst

def get_sentence_tokzer(cfg):
    '''
    grab a predefined sentence tokenizer
    '''
    with open(os.path.join(cfg['base_dir'], cfg['punkt_tok_src']), 'rb') as punk_fobj:
        trainer_params = pickle.load(punk_fobj)
        sent_tok = PunktSentenceTokenizer(trainer_params)
    logger.info(sent_tok._params.abbrev_types)
    return sent_tok

def gen_sent_tokzer(text_lst, cfg):
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

def get_pos_ind_dict(def_lst, cfg):
    # Finding the POS set 
    pos_cnt = Counter()
    for Def in def_lst:
        pos_cnt.update([el[0][1] for el in Def['ner']])
    print("Found {} alphanum POS tags in the data, the most common are: {}"\
          .format(len(pos_cnt), pos_cnt.most_common()[:10]))
    pos_lst = list(pos_cnt)
    pos_ind_dict = {pos: k for k, pos in enumerate(pos_lst)}
    cfg['Npos_cnt'] = len(pos_cnt)
    return  pos_ind_dict, cfg

def open_word_embedding(cfg):
    with open_w2v(os.path.join(cfg['base_dir'], cfg['embed_data_src'])) as embed_dict:
        wind = ['<UNK>',] + list(embed_dict.keys())
        cfg['emb_nvocab'] = len(wind) 
        embed_matrix = np.zeros((cfg['emb_nvocab'], 200))
        for word, vec in embed_dict.items():
            #vect = embed_dict.get(word)
            ind = wind.index(word)
                #vect = vect/np.linalg.norm(vect)
            embed_matrix[ind] = vec
    logger.info("Length of word index (wind) is {}".format(len(wind)))
    return wind, embed_matrix, cfg

def get_ranges_lst(def_lst, cfg):
    r_def_lst = range(len(def_lst))
    TVT_len = int(cfg['train_test_split']*len(def_lst))
    r_train = r_def_lst[:TVT_len]
    r_valid = r_def_lst[TVT_len:]
    r_test = r_valid[:int(0.5*len(r_valid))]
    r_valid = r_valid[int(0.5*len(r_valid)):]
    log_str = 'Original Range: {}\n Training: {}  Validation: {}  Test: {} \n'\
              .format(repr(r_def_lst), repr(r_train), repr(r_test), repr(r_valid)) 
    logger.info(log_str)

    train_def_lst = [def_lst[k] for k in r_train]
    test_def_lst = [def_lst[k] for k in r_test]
    valid_def_lst = [def_lst[k] for k in r_valid]

    return train_def_lst, test_def_lst, valid_def_lst

def prep_data(dat, wind, cfg, *args):
    '''
   dat should be in the "ner" format
    '''
    if isinstance(dat, str):
        dat_tok = word_tokenize(dat)
        norm_words = [clean.normalize_text(d).strip() for d in dat_tok]
        labels = [False for d in dat]
    else:
        norm_words = [clean.normalize_text(d[0][0]).strip() for d in dat]
        labels = [d[1] != 'O' for d in dat]
    ind_words = []
    for w in norm_words:
        try:
            ind_words.append(wind.index(w))
        except ValueError:
            ind_words.append(0)
    return ind_words, labels

def prep_pos(dat, pos_ind_dict):
    '''
    dat is in the format:
    [(('In', 'IN'), 'O'),
     (('Southern', 'NNP'), 'O'),
     (('Africa', 'NNP'), 'O'),
     ((',', ','), 'O'),
     (('the', 'DT'), 'O'),
     (('word', 'NN'), 'O')]
    '''
    out_lst = []
    for d in dat:
        out_lst.append(pos_ind_dict[d[0][1]])
    return out_lst

def binary_features(dat):
    out_lst = []
    for d in dat:
        word =d[0][0]
        capitalized = float(word[0] in string.ascii_uppercase)
        contains_dash = float('-' in word)
        
        out_lst.append((capitalized, contains_dash))
    return out_lst

def prep_data4real(_def_lst, wind, _pos_ind_dict, cfg):
    _data = [prep_data(d['ner'], wind, cfg) for d in _def_lst]
    _seq, _lab = zip(*_data)
    _pos_seq = [prep_pos(d['ner'], _pos_ind_dict) for d in _def_lst]
    _bin_seq = [binary_features(d['ner']) for d in _def_lst]
    # PAD THE SEQUENCES
    _seq = pad_sequences(_seq, **cfg['padseq'])
    _pos_seq = pad_sequences(_pos_seq, **cfg['padseq'])
    _bin_seq = pad_sequences(_bin_seq, **cfg['padseq'],
                                  value = cfg['nbin_feats']*[0.0],
                                 dtype='float32')
    _lab = pad_sequences(_lab, **cfg['padseq'])
    return _seq, _pos_seq, _bin_seq, _lab

def get_bilstm_lstm_model(embed_matrix, cfg_dict):
    model = Sequential()
    # Add Embedding layer
    model.add(Embedding(cfg_dict['input_dim'], 
                        output_dim=cfg_dict['output_dim'],
                        input_length=cfg_dict['input_length'],
                       weights = [embed_matrix],
                       trainable = False))
    #model.add(Embedding(cfg_dict['input_dim'], 
    #                    output_dim=cfg_dict['output_dim'],
    #                    input_length=cfg_dict['input_length']))
    # Add bidirectional LSTM
    model.add(Bidirectional(LSTM(units=cfg_dict['output_dim'],
                                 return_sequences=True,
                                 dropout=0.2, 
                                 recurrent_dropout=0.2), merge_mode = 'concat'))
    # Add LSTM
    model.add(Bidirectional(LSTM(units=cfg_dict['output_dim'],
                   return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                   recurrent_initializer='glorot_uniform'), merge_mode='concat'))
    # Add timeDistributed Layer
    model.add(TimeDistributed(Dense(1, activation="sigmoid")))
    #Optimiser 
    adam = Adam(**cfg_dict['adam'])
    # Compile model
    #bce = tf.keras.losses.BinaryCrossentropy(sample_weight=[0.3, 0.7])
    model.compile(loss = 'binary_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model

def bilstm_model_w_pos(embed_matrix, cfg_dict):
    
    words_in = Input(shape=(cfg_dict['input_length'], ), name='words-in')
    pos_in = Input(shape=(cfg_dict['input_length'], ), name='pos-in')
    bin_feats = Input(shape=(cfg_dict['input_length'], cfg_dict['nbin_feats']), name='bin-features-in')
    
    word_embed = Embedding(cfg_dict['input_dim'], 
                        output_dim=cfg_dict['output_dim'],
                        input_length=cfg_dict['input_length'],
                       weights = [embed_matrix],
                       trainable = False,
                          name='word-embed')(words_in)
    pos_embed = Embedding(cfg_dict['Npos_cnt'], 
                        output_dim=cfg_dict['pos_dim'],
                        input_length=cfg_dict['input_length'],
                          embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.),
                       trainable = True,
                         name='pos-embed')(pos_in)
    full_embed = Concatenate(axis=2)([word_embed, pos_embed, bin_feats])
    
    
    out = Bidirectional(LSTM(units=cfg_dict['lstm_units'],
                                 return_sequences=True,
                                 dropout=0.2, 
                                 recurrent_dropout=0.2),
                        merge_mode = 'concat')(full_embed)
    #out = GlobalMaxPooling1D(out) 
    # Add LSTM
    out = Bidirectional(LSTM(units=cfg_dict['lstm_units'],
                   return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                   recurrent_initializer='glorot_uniform'),
                        merge_mode = 'concat')(out)
    # Add timeDistributed Layer
    out = TimeDistributed(Dense(10, activation="relu"))(out)
    out = TimeDistributed(Dense(1, activation="sigmoid"))(out)
    #Optimiser 
    adam = Adam(**cfg_dict['adam'])
    # Compile model
    bce = tf.keras.losses.BinaryCrossentropy()  #(sample_weight=[0.3, 0.7])
    model = Model([words_in, pos_in, bin_feats], out)
    model.compile(loss = bce,   #'binary_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model

def train_model(X, y, t_seq, t_lab, model, cfg):
    # fit model for one epoch on this sequence
    res = model.fit(X, y, epochs=cfg['epochs'],
                    batch_size=cfg['batch_size'],
                    validation_data=(t_seq, t_lab),
                   callbacks=[])
    return res

def main():
    args = argParse()
    cfg = gen_cfg(args = args)

    logging.basicConfig(filename=os.path.join(cfg['save_path_dir'], 'training.log'),
            level=logging.INFO)
    logger.info("GPU devices: {}".format(list_physical_devices('GPU')))

    text_lst = get_wiki_pm_stacks_data(cfg)
    sent_tok = gen_sent_tokzer(text_lst, cfg)
    logger.info(sent_tok._params.abbrev_types)

    def_lst = ner.bio_tag.put_pos_ner_tags(text_lst, sent_tok)
    logger.info("Length of the def_lst is: {}".format(len(def_lst)))

    pos_ind_dict, cfg = get_pos_ind_dict(def_lst, cfg)

    wind, embed_matrix, cfg = open_word_embedding(cfg)

    train_def_lst, test_def_lst, valid_def_lst = get_ranges_lst(def_lst, cfg)

    train_seq, train_pos_seq, train_bin_seq , train_lab = prep_data4real(
            train_def_lst, wind, pos_ind_dict, cfg)
    test_seq, test_pos_seq, test_bin_seq , test_lab = prep_data4real(
            test_def_lst, wind, pos_ind_dict, cfg)
    valid_seq, valid_pos_seq, valid_bin_seq , valid_lab = prep_data4real(
            valid_def_lst, wind, pos_ind_dict, cfg)

    cfg.update({'input_dim': len(wind),
          'output_dim': 200, #don't keep hardcoding this
         'input_length': cfg['padseq']['maxlen'],
         'pos_dim': 3,
                'pos_constraint': 1/200,
         'n_tags': 2,
         'batch_size': 2000,
         'lstm_units': 150,
          'adam': {'lr': 0.025, 'beta_1': 0.9, 'beta_2': 0.999},
          'epochs': 70,})

    model_bilstm = bilstm_model_w_pos(embed_matrix, cfg)
    #history = train_model(train_seq, train_lab, test_seq, test_lab, model_bilstm_lstm, cfg )
    history = model_bilstm.fit([train_seq, train_pos_seq, train_bin_seq], train_lab, epochs=30,
                    batch_size=cfg['batch_size'],
                    validation_data=([test_seq, test_pos_seq, test_bin_seq], test_lab))


if __name__ == '__main__':
    main()

