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
try:
    import pickle5 as pickle
except ModuleNotFoundError:
    print('Module pickle5 not found. Continuing without it')
    import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional,\
                      GRU, Dropout, GlobalAveragePooling1D, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.config import list_physical_devices
from tensorflow.keras.callbacks import TensorBoard

import sys
sys.path.extend(["/home/luis/.local/lib/python3.8/site-packages"])

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
import classifier_models as M

# GET the Important Environment Paths
# the variables PERMSTORAGE and TEMPFASTSTORAGE are defined in the singularity container
base_dir = os.environ.get('PERMSTORAGE', '/media/hd1') 
mine_out_dir = os.environ.get('TEMPFASTSTORAGE', '/tmp/rm_me_dir')  # This is temporary fast storage
#data_dir = os.environ['DATA_DIR']  
# DATA_DIR is where the data that will be classified resides /media/hd1 or $LOCAL

os.makedirs(mine_out_dir, exist_ok=True)
tf_model_dir = os.path.join(base_dir,\
        'trained_models/lstm_classifier',\
        'lstm_Aug-19_04-15')

# PATH OF THE PROCESSED ARTICLES (directory: promath)
# singularity cannot reach env variables so have to use full paths
#data_path = '/opt/promath'
#data_path = os.path.join(data_dir, 'promath')
data_path = ''

# path to the training data
train_example_path = os.path.join(base_dir,
        'training_defs/math10/1009_004.xml.gz')

logging.basicConfig(filename=os.path.join(mine_out_dir, 'classifying.log'),
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
        #print(f"**Peeping into file {fname}  **")
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
    # dir_list has full paths
    opt_prob = float(cfg['opt_prob'])
    for k, dirname in enumerate(dir_lst):
    #for k, dirname in enumerate(['math' + repr(k)[2:] for k in range(1996, 1994, 1)]):
        logger.info('Classifying the contents of {}'.format(dirname))
        try:
            #full_path = os.path.join(mine_out_dir, cfg['promath_dir'], dirname)
            full_path = os.path.join(data_path, dirname)
            tar_lst = [os.path.join(full_path, p) for p in os.listdir(full_path)\
                    if '.tar.gz' in p]
        except FileNotFoundError:
            print(' %s Not Found'%full_path)
            break
        out_path = os.path.join(cfg['save_path'], os.path.basename(dirname))
        os.makedirs(out_path, exist_ok=True)
       
        for tfile in tar_lst:
            Now = dt.now()
            #clf = lstm_model
            vzer = Vectorizer()
            def_root = untar_clf_append(tfile, out_path,\
                    model, vzer, thresh=opt_prob)
            #print(etree.tostring(def_root, pretty_print=True).decode())
            gz_filename = os.path.basename(tfile).split('.')[0] + '.xml.gz' 
            print(gz_filename)
            gz_out_path = os.path.join(out_path, gz_filename) 
            class_time = (dt.now() - Now)
            Now = dt.now()
            #import pdb; pdb.set_trace()
            with gzip.open(gz_out_path, 'wb') as out_f:
                print("Writing to dfdum zipped file to: %s"%gz_out_path)
                out_f.write(etree.tostring(def_root, encoding='utf8', pretty_print=True))
            writing_time = (dt.now() - Now) 
            logger.info("Writing file to: {} CLASSIFICATION TIME: {} Writing Time {}"\
                             .format(gz_out_path, class_time, writing_time))

def mine_individual_file(filepath, cfg):
    '''
    Mines an individual tar.gz file from promath. More granular mining than `mine_dirs` 
    mines a xml.gz 
    filepath:
        Full path of the the tar.gz file to extract ex. /media/hd1/promath/math01/0103_001.tar.gz

    Saves to a directory with path cfg['save_path']/math01/0103_001.tar.gz
    '''
    opt_prob = float(cfg['opt_prob'])
    logger.info('Classifying the contents of {}'.format(filepath))
    try:
        # data_path is set globally to '' (empty string)
        # example filepath /media/hd1/promath/math01/0103_001.tar.gz
        full_path, tfile = os.path.split(filepath)
        dirname = os.path.split(full_path)[1]
        # expect full_path = /media/hd1/promath/math01
        # dirname = math01
        # tfile = 0103_001.tar.gz
        assert os.path.isdir(full_path), f"{full_path} is not a dir" 
        #assert os.path.isfile(tfile), f"File: {tfile} not found"
    except FileNotFoundError:
        print(' %s Not Found'%full_path)
    out_path = os.path.join(cfg['save_path'], dirname)
    os.makedirs(out_path, exist_ok=True)
   
    #for tfile in tar_lst:
    Now = dt.now()
    #clf = lstm_model
    vzer = Vectorizer()
    def_root = untar_clf_append(filepath, out_path,\
            model, vzer, thresh=opt_prob)
    #print(etree.tostring(def_root, pretty_print=True).decode())
    gz_filename = os.path.basename(tfile).split('.')[0] + '.xml.gz' 
    #print(gz_filename)
    gz_out_path = os.path.join(out_path, gz_filename) 
    class_time = (dt.now() - Now)
    Now = dt.now()
    #import pdb; pdb.set_trace()
    with gzip.open(gz_out_path, 'wb') as out_f:
        print("Writing to dfdum zipped file to: %s"%gz_out_path)
        out_f.write(etree.tostring(def_root, encoding='utf8', pretty_print=True))
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

def model_callback(cfg):
    '''
    Return a Tensorboard Callback
    '''
    return TensorBoard(log_dir=cfg['tboard_path'],
                            histogram_freq=1,
                            profile_batch="2,22")


def test_model(path, cfg):
    xml_lst = [path,]
    stream = stream_arxiv_paragraphs(xml_lst, samples=6000)
#os.path.join(base_dir,'training_defs/math10/1008_001.xml.gz'),
    all_data =[]
    Now1 = dt.now()
    for s in stream:
        all_data += list(zip(s[0], s[1]))
    logger.info('The length of the (test) all_data is {} the first element of xml_lst is: {}'\
            .format(len(all_data), xml_lst[0]))
    test = list(zip(*( all_data )))
    test_seq = [text2seq(t) for t in test[0]]
    test_seq = padding_fun(test_seq, cfg)
    prep_data_t = (dt.now() - Now1)

    Now2 = dt.now()

    #tboard_call = model_callback(cfg)
    ret = model.evaluate(test_seq, np.array(test[1]),
            )
            #callbacks=[tboard_call,])
    evaluation_t = (dt.now() - Now2)
    logger.info('TEST TIMES: prep data: {} secs -- evaluation: {} secs.'\
            .format(prep_data_t, evaluation_t))

    return ret

#########################
#### MAIN FUNCTION ######
#########################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='',
        help="Path to the trained model ex. '/media/hd1/trained_models/lstm_classifier/lstm_Aug-19_04-15")
    parser.add_argument('--out', type=str, default='',
        help="Path to dir 'mine_out_dir' to output mining results.")
    parser.add_argument('--mine', type=str, nargs='+',
            help='Path to data to mine, ex. /media/hd1/promath/math96')
    args = parser.parse_args()

    if args.model != '' :
        tf_model_dir = args.model

    if args.out != '' :
        mine_out_dir = args.out

    # GET THE PATH AND config
    cfg = open_cfg_dict(os.path.join(tf_model_dir, 'cfg_dict.json'))
    cfg['save_path'] = mine_out_dir
    cfg['tboard_path'] = os.path.join(mine_out_dir, 'tboard_logs') 
    idx2tkn, tkn2idx = open_idx2tkn_make_tkn2idx(os.path.join(tf_model_dir,\
            'idx2tkn.pickle'))
    print(tkn2idx['commutative'])
    
    cfg_model_type = cfg.get('model_type', None)
    if cfg_model_type == 'lstm':
        model = lstm_model_one_layer(cfg)
    elif cfg_model_type == 'conv':
        model = M.conv_model_globavgpool(cfg, logger)
    else:
        logger.info("cfg['model_type'] could not be found, assuming lstm_one_layer")
        model = lstm_model_one_layer(cfg)
        
    # READ THE MODEL AND LOAD WEIGHTS
    model.load_weights(tf_model_dir + '/model_weights')
    logger.info("CONFIG cfg = {}".format(cfg))

    # TEST
    test_result = test_model(train_example_path, cfg)
    logger.info(f'TEST Loss: {test_result[0]:1.3f} and Accuracy: {test_result[1]:1.3f}')

    if args.mine is not None:
        logger.info('List of Mining dirs: {}'.format(args.mine))
        #mine_dirs(args.mine, cfg)
        mine_individual_file(args.mine[0], cfg)
    else:
        logger.info('--mine is empty there will be no mining.')


