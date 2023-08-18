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
import json
import logging
from datetime import datetime as dt
#import train_ner as TN
from functools import reduce

import sklearn.metrics as metrics
#import matplotlib.pyplot as plt

from nltk.chunk import conlltags2tree, tree2conlltags

import gzip
import glob
from lxml import etree
from tqdm import tqdm
import random
from collections import Counter

import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
#from unwiki import unwiki
import ner
from embed_utils import open_w2v
import clean_and_token_text as clean

# SETUP LOGGING
#logging.basicConfig(filename=os.path.join('/tmp/trainer', 'ner_inference.log'),
#        level=logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.info("GPU devices: {}".format(list_physical_devices('GPU')))

def bilstm_lstm_model_w_pos(cfg_dict):
    words_in = Input(shape=(cfg_dict['input_length'], ), name='words-in')
    pos_in = Input(shape=(cfg_dict['input_length'], ), name='pos-in')
    bin_feats = Input(shape=(cfg_dict['input_length'], cfg_dict['nbin_feats']),
            name='bin-features-in')
    
    word_embed = Embedding(cfg_dict['input_dim'], 
                        output_dim=cfg_dict['output_dim'],
                        input_length=cfg_dict['input_length'],
                       #weights = [embed_matrix],
                       trainable = False,
                          name='word-embed')(words_in)
    pos_embed = Embedding(cfg['Npos_cnt'], 
                        output_dim=cfg_dict['pos_dim'],
                        input_length=cfg_dict['input_length'],
                          embeddings_initializer=tf.keras.initializers.RandomNormal(
                              mean=0.,
                              stddev=1.),
                           trainable = True,
                         name='pos-embed')(pos_in)
    full_embed = Concatenate(axis=2)([word_embed, pos_embed, bin_feats])
    
    
    out = Bidirectional(LSTM(units=330, 
                                 return_sequences=True,
                                 dropout=0.2, 
                                 recurrent_dropout=0.2,
                                 recurrent_initializer='orthogonal'),
                        merge_mode = 'concat')(full_embed)
    #out = GlobalMaxPooling1D(out) 
    # Add LSTM
    out = Bidirectional(LSTM(units=cfg['lstm_units'],
                   return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                   recurrent_initializer='orthogonal'),
                        merge_mode = 'concat')(out)
    # Add timeDistributed Layer
    out = TimeDistributed(Dense(10, activation="relu"))(out)
    out = TimeDistributed(Dense(1, activation="sigmoid"))(out)
    #Optimiser 
    adam = Adam(**cfg['adam'])
    # Compile model
    bce = tf.keras.losses.BinaryCrossentropy()  #(sample_weight=[0.3, 0.7])
    model = Model([words_in, pos_in, bin_feats], out)
    model.compile(loss = bce,   #'binary_crossentropy',
                  optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model

def open_cfg_dict(path):
    with open(path, 'r') as cfg_fobj:
        cfg = json.load(cfg_fobj)
    return cfg

def read_word_index_tkn2idx(path):
    global wind
    global tkn2idx
    with open(path, 'rb') as f:
        wind = pickle.load(f)
    tkn2idx = {tok: idx for idx, tok in enumerate(wind)}
    return wind, tkn2idx

def read_pos_index_pos2idx(path):
    global pos_lst
    global pos_ind_dict
    with open(path, 'rb') as f:
        pos_lst = pickle.load(f)
    pos_ind_dict = {tok: idx for idx, tok in enumerate(pos_lst)}
    return pos_lst, pos_ind_dict

def read_sent_tok(path):
    with open(path, 'rb') as punk_fobj:
        trainer_params = pickle.load(punk_fobj)
        sent_tok = PunktSentenceTokenizer(trainer_params)
    return sent_tok

def split_fields(elem):
    # this function is just used in the planetmath data in test_model
    title = elem.find('.//dfndum').text 
    section = elem.get('name')
    defin = elem.find('.//stmnt').text
    return (title, section, defin)

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

    
def prep_data4real(_def_lst, wind, pos_ind_dict, cfg):
    _data = [prep_data(d['ner'], wind, cfg) for d in _def_lst]
    _seq, _lab = zip(*_data)
    _pos_seq = [prep_pos(d['ner'], pos_ind_dict) for d in _def_lst]
    _bin_seq = [binary_features(d['ner']) for d in _def_lst]
    # PAD THE SEQUENCES
    _seq = pad_sequences(_seq, **cfg['padseq'])
    _pos_seq = pad_sequences(_pos_seq, **cfg['padseq'])
    _bin_seq = pad_sequences(_bin_seq, **cfg['padseq'],
                                  value = cfg['nbin_feats']*[0.0],
                                 dtype='float32')
    _lab = pad_sequences(_lab, **cfg['padseq'])
    return _seq, _pos_seq, _bin_seq, _lab

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

def get_chunkscore(test_def_lst, preds, CutOFF):
    test_pred_lst = switch_to_pred(test_def_lst, preds, cutoff=CutOFF)
    unpack = lambda l: [(tok, pos, ner) for ((tok, pos), ner) in l]
    Tree_lst_gold = [conlltags2tree(unpack(t['ner'])) for t in test_def_lst]
    Tree_lst_pred = [conlltags2tree(unpack(t)) for t in test_pred_lst]

    chunkscore = ChunkScore()
    for i in range(len(Tree_lst_gold)):
        chunkscore.score(Tree_lst_gold[i], Tree_lst_pred[i])
    return chunkscore

def test_model(model, sent_tok, wind, pos_ind_dict, cfg):
    plmath = []
    #test_data = os.path.join(cfg['base_dir'],
    #        'planetmath/datasets/planetmath_definitions.xml.gz')
    test_data = '/opt/data_dir/planetmath/datasets/planetmath_definitions.xml.gz'
    with gzip.open(test_data, 'r') as xml_fobj:
        def_xml = etree.parse(xml_fobj)
        for art in def_xml.findall('article'):
            plmath.append(split_fields(art))
    text_lst = plmath
    random.shuffle(text_lst)
    def_lst = ner.bio_tag.put_pos_ner_tags(text_lst, sent_tok)
    print('There are {} definitions.'.format(len(text_lst)))
    test_seq, test_pos_seq, test_bin_seq , test_lab = prep_data4real(
        def_lst, wind, pos_ind_dict, cfg)
    preds = model.predict([test_seq, test_pos_seq, test_bin_seq])
    model.evaluate([test_seq, test_pos_seq, test_bin_seq], test_lab)
    logging.getLogger().info(get_chunkscore(def_lst, preds, cfg['tboy']['cutoff']))
    print(get_chunkscore(def_lst, preds, cfg['tboy']['cutoff']))


# 
def add_dfndum(D, term):
    # Add term to definition element D in a dfndum tag
    dfndum = etree.SubElement(D, 'dfndum')
    dfndum.text = term
    return D

def str_tok_pos_tags(defp, tok):
    '''
    INPUTS
    ------
    defp: a string representing a Paragraph of a definition
          with possible many sentences.
    EX.
    defl: A curve _inline_math_ is said to be
    output format:
               [ [ ('A', 'DT'),
                ('curve', 'NN'),
                ('_inline_math_', 'NN'),
                ('is', 'VBZ'),
                ('said', 'VBD'),
                ('to', 'TO'),
    '''
    if not isinstance(defp, str):
        raise ValueError('str_tok_pos_tags Only takes str as defl.')
        
    big_lst = []
    for d in tok.tokenize(defp):
        pos_tokens = pos_tag(word_tokenize(d))
        big_lst.append(pos_tokens)
    return big_lst

def apply_wind_pos_ind(tag_str, word_dict, pos_dict):
    '''
    input format:
               [ [ None or parent_def,
               ('A', 'DT'),
                ('curve', 'NN'),
                ('_inline_math_', 'NN'),
                ('is', 'VBZ'),
                ('said', 'VBD'),
                ('to', 'TO'), ]]
                
    output format:
          [[ (12, 4, 1.0, 0.0),
          (2159, 1, 0.0, 0.0),
          (5, 19, 0.0, 0.0),
          (3611, 1, 0.0, 0.0),
          (7, 2, 0.0, 0.0),
    '''
        # Normalize each word and get word index or default 0
    index_word = lambda w: word_dict.get(clean.normalize_text(w).strip(), 0)
        # Binary data
    capitalized = lambda w: float(w[0] in string.ascii_uppercase)
    contains_dash = lambda word: float('-' in word)

    output_lst = []
    for sent in tag_str:
        norm_words = [ ( index_word(d[0]), pos_dict[d[1]],
                       capitalized(d[0]), contains_dash(d[0]))  for d in sent]
        output_lst.append(norm_words)
    return output_lst

def pad_unpack_stack(pos_sents, cfg):
    w_seq = []
    p_seq = []
    b_seq = []
    for sent in pos_sents:
        _seq, _pos_seq, *_bin_seq = list(zip(*sent))
        w_seq.append(_seq)
        p_seq.append(_pos_seq)
        b_seq.append(_bin_seq)
        
    b_seq = [list(zip(*l)) for l in b_seq] # unpack b_seq to list of binary pairs
    w_seq = pad_sequences(w_seq, **cfg['padseq'])
    p_seq = pad_sequences(p_seq, **cfg['padseq'])
    b_seq = pad_sequences(b_seq, **cfg['padseq'],
                              value = cfg['nbin_feats']*[0.0],
                             dtype='float32')
    return w_seq, p_seq, b_seq

def _prep_raw_data(text, tok, word_dict, pos_dict, cfg):
    # just a function for other function to wrap around
    pos_sents = str_tok_pos_tags(text, tok)

    pos_sents = apply_wind_pos_ind(pos_sents, tkn2idx, pos_ind_dict)

    return pad_unpack_stack(pos_sents, cfg) # returns words, pos, binary

def crop_term_words(sent, P):
    '''
    sent has the format:
    [('_display_math_', 'IN'),
 ('The', 'DT'),
 ('Ursell', 'NNP'),
 ('functions', 'NNS'),
 ('of', 'IN'),
 ('a', 'DT'),
 ('single', 'JJ'),
     
    '''
    term_lst = []
    k = 0
    while True:
        try:
            assert isinstance(P[k], np.bool_), f"P={P} is not a numpy bool"
            if P[k]:
                term_str = sent[k][0]
                k += 1
                while P[k]:
                    term_str += ' ' + sent[k][0]
                    k += 1
                term_lst.append(term_str)
            else:
                k += 1
        except IndexError:
            break
    return term_lst
            
    
def prep_raw_data_and_mine(xml_path, tok, word_dict, pos_dict, cfg, model=None):
    '''
    xml_path: path to a compressed xml file of definitions
    
    tok: sentence tokenizer
    
    word_dict: dictionary word -> index
    
    pos_dict: dictionary pos_code -> index
    ---------------------------
    Returns:
    Data ready for the model: word_seq (N,50), pos_seq (N,50), bin_seq (N,50,2)
    def_indices: list with format [(def_tag, [0,1,2,3]), ]
    '''
    if isinstance(xml_path, str):
        if not os.path.isfile(xml_path): raise ValueError(f"{xml_path} is Not a file")
        pars = etree.XMLParser(recover=True)
        xml_tree = etree.parse(xml_path, parser=pars)
        root = xml_tree.getroot()
        Defs = root.findall('.//definition')
    
    def_indices = []
    prev_ind = 0
    word_seqs = np.zeros([0, cfg['padseq']['maxlen']])
    pos_seqs = np.zeros([0, cfg['padseq']['maxlen']])
    bin_seqs = np.zeros([0, cfg['padseq']['maxlen'], cfg['nbin_feats']])
    all_word_sents = []
    for D in Defs:
        text = D.find('stmnt').text
        word_sents = str_tok_pos_tags(text, tok)
        all_word_sents += word_sents

        indexed_sents = apply_wind_pos_ind(word_sents, tkn2idx, pos_ind_dict)
        wo, po, bi = pad_unpack_stack(indexed_sents, cfg) 
        # returns words, pos, binary_prep_raw_data(text, tok, word_dict, pos_dict, cfg)
        
        word_seqs = np.append(word_seqs, wo, axis=0)
        pos_seqs = np.append(pos_seqs, po, axis=0)
        bin_seqs = np.append(bin_seqs, bi, axis=0)
        
        def_indices.append((D, range(prev_ind, prev_ind+len(word_sents))))
        prev_ind += len(word_sents)
    
    # Invert def_indices
    inv_indices = len(all_word_sents)*[None]
    for e in def_indices:
        for k in e[1]:
            inv_indices[k] = e[0]
    assert all([not(l == None) for l in inv_indices])
    
    # INFERENCE AND REGISTRATION
    if model is not None and len(all_word_sents) > 0:
        preds = model.predict([word_seqs, pos_seqs, bin_seqs])
        assert len(all_word_sents) == preds.shape[0]
        for i in range(len(all_word_sents)):
            term_lst = crop_term_words(all_word_sents[i],
                        [p[0] for p in (preds[i]>cfg['tboy']['cutoff'])])
            for Term in term_lst:
                add_dfndum(inv_indices[i], Term)
    else:
        preds = None
    
    #with open('/home/luis/rm_me.xml', 'w') as xml_fobj:
    #    xml_fobj.write(etree.tostring(root, pretty_print=True).decode('utf8'))
    
    return root #word_seqs, pos_seqs, bin_seqs, def_indices

def mine_individual_file(fname_, sent_tok, tkn2idx, pos_ind_dict, cfg, model=None):
    '''
    fname_ is the full path of an .xml.gz file with all the definitions classified 
    eg. /opt/data_dir/glossary/inference_class_all/math96/9601_001.xml.gz
    '''
    print(f"Mining files: {fname_}")
    t3 = dt.now()
    basename = os.path.basename(fname_)
    dirname = os.path.dirname(fname_)
    mathname = os.path.basename(dirname) # ej. math10
    logger.info(f"Files for mining: {fname_}, {basename}") 
    out_root = prep_raw_data_and_mine(fname_,
            sent_tok, tkn2idx,
            pos_ind_dict,
            cfg, model=model)

    out_dir_math = os.path.join(cfg['outdir'], mathname)
    os.makedirs(out_dir_math, exist_ok=True)
    with gzip.open( os.path.join(out_dir_math, basename), 'wb') as out_fobj:
        out_fobj.write(etree.tostring(out_root, encoding='utf8', pretty_print=True))
    logger.info("Inference time on {}: {}".format(basename, (dt.now() - t3)))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mine', type=str, nargs='+',
            default='/media/hd1/glossary/inference_class_all/',
            help='Path to data to be mined')
    parser.add_argument('--model', type=str,
            default='/home/luis/ner_model',
            help='Path to the tensorflow model directory')
    parser.add_argument('--out', type=str,
            default='/home/luis/NNglossary',
            help='Local path to save')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    # GET THE PATH AND config
    tf_model_dir = args.model
    cfg = open_cfg_dict(os.path.join(tf_model_dir, 'cfg.json'))
    cfg.update({'outdir': args.out})
    logger.info(repr(cfg))
    # GET WORD INDICES
    wind, tkn2idx = read_word_index_tkn2idx(os.path.join(tf_model_dir,\
            'wordindex.pickle'))
    pos_lst, pos_ind_dict = read_pos_index_pos2idx(os.path.join(tf_model_dir,\
            'posindex.pickle'))
    logger.info("index of commutative is: {}".format(tkn2idx['commutative']))
    logger.info("POS index of VBD is {}".format(pos_ind_dict['VBD']))
    # GET THE SENTENCE TOKENIZER
    sent_tok = read_sent_tok(os.path.join(tf_model_dir, 'punkt_params.pickle'))
    logger.info(sent_tok._params.abbrev_types)

    # LOAD TF MODEL
    #model = bilstm_lstm_model_w_pos(cfg)
    #model = TN.bilstm_with_pos(cfg)
    #model.load_weights(os.path.join(tf_model_dir, 'bilstm_with_pos'))
    model = tf.keras.models.load_model(
            os.path.join(tf_model_dir, 'bilstm_with_pos'))

    # TEST
    t1 = dt.now()
    test_model(model, sent_tok, wind, pos_ind_dict, cfg)
    logger.info("Testing time: {}".format((dt.now() - t1)))

    # INFERENCE
    logger.info("Starting inference")
    t2 = dt.now()
    #fname = args.mine+'math05/0509_001.xml.gz'
    mine_lst = args.mine
    print(f"Mining files: {mine_lst}")
    for fname in mine_lst:
        mine_individual_file(fname,
                  sent_tok, tkn2idx,
                  pos_ind_dict,
                  cfg, model=model)
    logger.info("TOTAL INFERENCE TIME: {}".format((dt.now() - t2)))

if __name__ == "__main__":
    main()
