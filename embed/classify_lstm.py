from glob import glob
import os
import numpy as np
from lxml import etree
from collections import Counter
from random import shuffle
from datetime import datetime as dt
import logging
import gzip
import json
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional,\
                      GRU, Dropout, GlobalAveragePooling1D, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.config import list_physical_devices


import sklearn.metrics as metrics
# -

# %load_ext autoreload
# %autoreload 2
from embed_utils import open_w2v
from clean_and_token_text import normalize_text
import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from classifier_trainer.trainer import stream_arxiv_paragraphs
import parsing_xml as px
from extract import Definiendum
import peep_tar as peep

# GET the Important Environment Paths
base_dir = os.environ['PROJECT'] # This is permanent storage
local_dir = os.environ['LOCAL']  # This is temporary fast storage

main_path = os.path.join(base_dir,\
        'trained_models/lstm_classifier',\
        'lstm_Feb-21_19-12')

logging.basicConfig(filename=os.path.join(main_path, 'classifying.log'),
        level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("GPU devices: {}".format(list_physical_devices('GPU')))

def open_cfg_dict(path):
    with open(path, 'r') as cfg_fobj:
        cfg = json.load(cfg_fobj)
    return cfg

#Define these variables Globally
idx2tkn = ()
tkn2idx = {}

def open_idx2tkn_make_tkn2idx(path):
    global idx2tkn 
    global tkn2idx
    with open(path, 'rb') as f:
        idx2tkn = pickle.load(f)
    tkn2idx = {tok: idx for idx, tok in enumerate(idx2tkn)}
    return idx2tkn, tkn2idx

def text2seq(text):
    if type(text) == str:
        text = normalize_text(text).split()
    return [tkn2idx.get(s, 0) for s in text]

def padding_fun(seq, cfg):
    # Apply pad_sequence using the cfg dictionary
    return pad_sequences(seq, maxlen=cfg['max_seq_len'],
                            padding='post', 
                            truncating='post',
                            value=tkn2idx['�']) 

######################################################
# START THE CLASSIFICATION OF ARTICLE PARAGRAPHS     #
######################################################

class Vectorizer():
    def __init__(self):
        pass
    def transform(self, L):
        return padding_fun([text2seq(d) for d in L], cfg)

def untar_clf_append(tfile, out_path, clf, vzer, thresh=0.5, min_words=15):
    '''
    Arguments:
    `tfile` tarfile with arxiv format ex. 1401_001.tar.gz
    `out_path` directory to save the xml.tar.gz file with the same name as `tfile`
    `clf` model with .predict() attribute
    `vzer` funtion that take the text of a paragraph and outputs padded np.arrays for `clf`
    '''
    opt_prob = float(cfg['opt_prob'])
    root = etree.Element('root')
    for fname, tar_fobj in peep.tar_iter(tfile, '.xml'):
        try:
            DD = px.DefinitionsXML(tar_fobj) 
            if DD.det_language() in ['en', None]:
                art_tree = Definiendum(DD, clf, None, vzer,\
                        None, fname=fname, thresh=opt_prob,\
                        min_words=cfg['min_words']).root
                if art_tree is not None: root.append(art_tree)
        except ValueError as ee:
            print(f"{repr(ee)}, 'file: ', {fname}, ' is empty'")
    return root
    
def mine_dirs(dir_lst, cfg):
    opt_prob = float(cfg['opt_prob'])
    for k, dirname in enumerate(dir_lst):
    #for k, dirname in enumerate(['math' + repr(k)[2:] for k in range(1996, 1994, 1)]):
        logger.info('Classifying the contents of {}'.format(dirname))
        try:
            full_path = os.path.join(base_dir, cfg['promath_dir'], dirname)
            tar_lst = [os.path.join(full_path, p) for p in os.listdir(full_path)\
                    if '.tar.gz' in p]
        except FileNotFoundError:
            print(' %s Not Found'%full_path)
            break
        out_path = os.path.join(local_dir, cfg['save_path'], dirname)
        os.makedirs(out_path, exist_ok=True)
       
        for tfile in tar_lst:
            Now = dt.now()
            #clf = lstm_model
            vzer = Vectorizer()
            def_root = untar_clf_append(tfile, out_path,\
                    lstm_model, vzer, thresh=opt_prob)
            #print(etree.tostring(def_root, pretty_print=True).decode())
            gz_filename = os.path.basename(tfile).split('.')[0] + '.xml.gz' 
            print(gz_filename)
            gz_out_path = os.path.join(out_path, gz_filename) 
            class_time = (dt.now() - Now)
            Now = dt.now()
            with gzip.open(gz_out_path, 'wb') as out_f:
                print("Writing to dfdum zipped file to: %s"%gz_out_path)
                out_f.write(etree.tostring(def_root, pretty_print=True))
            writing_time = (dt.now() - Now) 
            logger.info("Writing file to: {} CLASSIFICATION TIME: {} Writing Time {}"\
                             .format(gz_out_path, class_time, writing_time))

#mine_dirs(['math96',])

def lstm_model_one_layer(cfg):
    lstm_model = Sequential([
        Embedding(cfg['tot_words'], cfg['embed_dim'], 
                  input_length=cfg['max_seq_len'],# weights=[embed_matrix],
                  trainable=False),
        Bidirectional(LSTM(cfg['lstm_cells'], return_sequences=True)),
        GlobalAveragePooling1D(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_model.summary(print_fn=logger.info) 
    return lstm_model

def test_model(path):
    xml_lst = [path,]
    stream = stream_arxiv_paragraphs(xml_lst, samples=6000)
#os.path.join(base_dir,'training_defs/math10/1008_001.xml.gz'),
    all_data =[]
    for s in stream:
        all_data += list(zip(s[0], s[1]))
    print('The length of all_data is {} the first element of xml_lst is: {}'\
            .format(len(all_data), xml_lst[0]))
    test = list(zip(*( all_data )))
    test_seq = [text2seq(t) for t in test[0]]
    test_seq = padding_fun(test_seq, cfg)
    ret = lstm_model.evaluate(test_seq, np.array(test[1]))
    return ret

#########################
#### MAIN FUNCTION ######
#########################
if __name__ == '__main__':
    # GET THE PATH AND DATA SOMEHOW
    cfg = open_cfg_dict(os.path.join(main_path, 'cfg_dict.json'))
    idx2tkn, tkn2idx = open_idx2tkn_make_tkn2idx(os.path.join(main_path,\
            'idx2tkn.pickle'))
    print(tkn2idx['commutative'])
    
    lstm_model = lstm_model_one_layer(cfg)
    lstm_model.load_weights(main_path + '/model_weights')
    test_result = test_model(base_dir + '/training_defs/math10/1009_004.xml.gz')
    logger.info(f'TEST Loss: {test_result[0]:1.3f} and Accuracy: {test_result[1]:1.3f}')

    mine_dirs(['math03'], cfg)

