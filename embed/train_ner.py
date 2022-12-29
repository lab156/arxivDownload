from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional,\
                      GRU, Dropout,GlobalAveragePooling1D, Conv1D, TimeDistributed,\
                      Input, Concatenate, GlobalMaxPooling1D
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
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
import json
from collections import Counter

import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from unwiki import unwiki
import ner
from embed_utils import open_w2v
import clean_and_token_text as clean
import train_utils as Tut


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

    cfg.update({'input_dim': None,
          'output_dim': None, #This need to be set when the embedding is opened
         'input_length': cfg['padseq']['maxlen'],
         'pos_dim': 3,
                'pos_constraint': 1/200,
         'n_tags': 2,
         'batch_size': 2000,
         'lstm_units1': 150,
         'lstm_units2': 150,
          'adam': {'lr': 0.025, 'beta_1': 0.9, 'beta_2': 0.999},
          'epochs': 35,})

    return cfg

def argParse():
    '''
    Parsing all arguments
    '''
    import argparse
    parser = argparse.ArgumentParser(description="Train LSTM model")
    parser.add_argument('--epochs', type=int, default=10,
            help="Number of epochs to train. Overrides default value")
    parser.add_argument('--experiments', type=int, default=1,
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
        for art in def_xml.findall('article'):
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
    return sent_tok, trainer.get_params()

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
    return  pos_ind_dict, pos_lst, cfg

def open_word_embedding(cfg):
    embed_dict = open_w2v(os.path.join(cfg['base_dir'], cfg['embed_data_src'])) 
    wind = ['<UNK>',] + list(embed_dict.keys())
    cfg['emb_nvocab'] = len(wind) 
    cfg['input_dim'] = len(wind)

    cfg['output_dim'] = embed_dict['a'].shape[0] #choose a common word

    embed_matrix = np.zeros((cfg['emb_nvocab'], cfg['output_dim']))
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
    model.add(Bidirectional(LSTM(units=cfg_dict['output_dim'], # this looks weird
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
    bin_feats = Input(shape=(cfg_dict['input_length'], cfg_dict['nbin_feats']),
            name='bin-features-in')
    
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
    
    
    out = Bidirectional(LSTM(units=cfg_dict['lstm_units1'],
                                 return_sequences=True,
                                 dropout=0.2, 
                                 recurrent_dropout=0.2,),
                        merge_mode = 'concat')(full_embed)
    #out = GlobalMaxPooling1D(out) 
    # Add LSTM
    out = Bidirectional(LSTM(units=cfg_dict['lstm_units2'],
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

def model_callbacks(cfg):
    '''
    Return a Tensorboard Callback
    '''
    cb = [] #
    if cfg['profiling'] == True:
        cb.append(TensorBoard(log_dir=cfg['prof_dir'],
                                histogram_freq=1,
                                profile_batch="2,22"))

    if 'mon_val_loss' in cfg['callbacks']:
        save_checkpoint = ModelCheckpoint(
                 os.path.join(cfg['save_path_dir'], 'model_saved'), \
                monitor='val_accuracy', verbose=1, \
                save_best_only=True, save_weights_only=False, \
                mode='max', save_freq='epoch')
        cb.append(save_checkpoint)

    if 'epoch_times' in cfg['callbacks']:
        ep_times = Tut.TimeHistory()
        cb.append(ep_times)
    else:
        ep_times = []

    if 'ls_schedule' in cfg['callbacks']:
        lr_sched = LearningRateScheduler(
                Tut.def_scheduler(cfg['AdamCfg']['lr_decay']))
        cb.append(lr_sched)

    if 'early_stop' in cfg['callbacks']:
        early = EarlyStopping(monitor='val_accuracy',
                patience=6,
                restore_best_weights=True)
        cb.append(early)

    return cb, ep_times

def train_model(X, y, t_seq, t_lab, model, cfg):
    # fit model for one epoch on this sequence
    res = model.fit(X, y, epochs=cfg['epochs'],
                    batch_size=cfg['batch_size'],
                    validation_data=(t_seq, t_lab),
                   callbacks=[])
    return res

def tf_bio_tagger(tf_pred, tag_def='DFNDUM', tag_o = 'O'):
    '''
    Convert a T/F (binary) Sequence into a BIO tag sequence
    [True, False, False] -> [B-DFNDUM, O, O]
    '''
    begin_tag = 'B-' + tag_def
    inside_tag = 'I-' + tag_def
    out_tag = tag_o
    return_tags = []
    for ind, x in enumerate(tf_pred):
        if x:
            if ind > 0:
                ret = inside_tag if tf_pred[ind - 1] else begin_tag
                return_tags.append(ret)
            else:
                return_tags.append(begin_tag)
        else:
            return_tags.append(out_tag)
    return return_tags
        
def switch_to_pred(test_def_lst, preds, cutoff = 0.5):
    #if len(preds.shape) == 2:
        #case just one prediction (50, 1) 
    #    preds = [preds]
    out_lst = []
    for k, pred in enumerate(preds):
        tf_pred = (pred > cutoff)
        test_def = test_def_lst[k]['ner']
        bio_pred = tf_bio_tagger(tf_pred)
        switched_def_lst = []
        for i in range(len(bio_pred)):    #rate(bio_pred):
            try:
                # test_def[i] example: (('Fock', 'NNP'), 'B-DFNDUM')
                tok_pos = test_def[i][0]
                switched_def_lst.append((tok_pos, bio_pred[i]))
            except IndexError:
                break
        out_lst.append(switched_def_lst)
    return out_lst

def get_chunkscore(CutOFF, test_def_lst, preds):
    test_pred_lst = switch_to_pred(test_def_lst, preds, cutoff=CutOFF)
    unpack = lambda l: [(tok, pos, ner) for ((tok, pos), ner) in l]
    Tree_lst_gold = [conlltags2tree(unpack(t['ner'])) for t in test_def_lst]
    Tree_lst_pred = [conlltags2tree(unpack(t)) for t in test_pred_lst]

    chunkscore = ChunkScore()
    for i in range(len(Tree_lst_gold)):
        chunkscore.score(Tree_lst_gold[i], Tree_lst_pred[i])
    return chunkscore
 
def tboy_finder(model, test_seq, test_pos_seq, test_bin_seq, test_lab, test_def_lst, cfg):

    preds = model.predict([test_seq, test_pos_seq, test_bin_seq])

    #data_points = []
    BOY_f_score = (0, 0) # (CutOff, score)
    for co in np.arange(0.1, 1, 0.1):
        cs = get_chunkscore(co, test_def_lst, preds)
        #data_points.append((cs.accuracy(), cs.f_measure()))
        if cs.f_measure() > BOY_f_score[1]:
            BOY_f_score = (co, cs.f_measure())

    logger.info(get_chunkscore(BOY_f_score[0], test_def_lst, preds))
    logger.info(f"Cutoff:  {BOY_f_score[0]}")

    loss, acc = model.evaluate([test_seq, test_pos_seq, test_bin_seq], test_lab)
    logger.info('Evaluate results: loss = {}   acc = {}'.format(loss, acc))
    cfg['tboy'] = {'cutoff': BOY_f_score[0], 'f_meas': BOY_f_score[1]}

    return BOY_f_score, cfg

def save_cfg_data(model, cfg, wind, pos_lst, trainer_params):
    # SAVE AND LOAD MODELS WEIGHTS
    model.save(os.path.join(cfg['save_path_dir'], 'bilstm_with_pos'))
    with open(os.path.join(cfg['save_path_dir'], 'cfg.json'), 'w') as cfg_fobj:
        json.dump(cfg, cfg_fobj)
    with open(os.path.join(cfg['save_path_dir'],'wordindex.pickle'), 'wb') as wind_fobj:
        pickle.dump(wind, wind_fobj) 
    with open(os.path.join(cfg['save_path_dir'],'posindex.pickle'), 'wb') as pos_fobj:
        pickle.dump(pos_lst, pos_fobj)
    with open(os.path.join(cfg['save_path_dir'], 'punkt_params.pickle'), 'wb') as punkt_fobj:
        pickle.dump(trainer_params, punkt_fobj)

def main():
    args = argParse()
    cfg = gen_cfg(args = args)

    logging.basicConfig(filename=os.path.join(cfg['save_path_dir'], 'training.log'),
            level=logging.INFO)
    logger.info("GPU devices: {}".format(list_physical_devices('GPU')))

    # RETRIEVE AND PREP DATA -----------------------------------------------
    text_lst = get_wiki_pm_stacks_data(cfg)
    sent_tok, trainer_params = gen_sent_tokzer(text_lst, cfg)
    logger.info(sent_tok._params.abbrev_types)

    def_lst = ner.bio_tag.put_pos_ner_tags(text_lst, sent_tok)
    random.shuffle(def_lst)
    logger.info("Length of the def_lst is: {}".format(len(def_lst)))

    pos_ind_dict, pos_lst, cfg = get_pos_ind_dict(def_lst, cfg)

    wind, embed_matrix, cfg = open_word_embedding(cfg)

    train_def_lst, test_def_lst, valid_def_lst = get_ranges_lst(def_lst, cfg)

    train_seq, train_pos_seq, train_bin_seq , train_lab = prep_data4real(
            train_def_lst, wind, pos_ind_dict, cfg)
    test_seq, test_pos_seq, test_bin_seq , test_lab = prep_data4real(
            test_def_lst, wind, pos_ind_dict, cfg)
    valid_seq, valid_pos_seq, valid_bin_seq , valid_lab = prep_data4real(
            valid_def_lst, wind, pos_ind_dict, cfg)

    # TRAIN AND DEFINE MODEL ---------------------------------------------
    model_bilstm = bilstm_model_w_pos(embed_matrix, cfg)
    #history = train_model(train_seq, train_lab, test_seq, test_lab, model_bilstm_lstm, cfg )
    history = model_bilstm.fit([train_seq, train_pos_seq, train_bin_seq], train_lab, 
            epochs=cfg['epochs'],
            batch_size=cfg['batch_size'],
                    validation_data=([valid_seq, valid_pos_seq, valid_bin_seq], valid_lab))

    boy_f, cfg = tboy_finder(model_bilstm, test_seq, 
            test_pos_seq, test_bin_seq, test_lab, test_def_lst, cfg)

    # SAVING DATA AND SOMETIMES MODELS ---------------------------------------
    save_cfg_data(model_bilstm, cfg, wind, pos_lst, trainer_params)


if __name__ == '__main__':
    main()

