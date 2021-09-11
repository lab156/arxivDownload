from glob import glob
import os
import numpy as np
from lxml import etree
from collections import Counter
from random import shuffle
from datetime import datetime as dt
import logging
logger = logging.getLogger(__name__)
import gzip
import json
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional,\
                      GRU, Dropout, GlobalAveragePooling1D, Conv1D
from tensorflow.keras.models import Sequential

from tensorflow.config import list_physical_devices
from tensorflow.keras.callbacks import TensorBoard


import sklearn.metrics as metrics

import sys
# THIS IS NEEDED TO RUN AS ROOT 
sys.path.extend(["/home/luis/.local/lib/python3.8/site-packages"])
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




def gen_cfg(*args, **kwargs):
    # GET the default values
    # create save_path_dir and profiling directory
    cfg = {'batch_size': 5000,
          'glob_data_source': '/training_defs/math*/*.xml.gz',
          'TVT_split' : 0.8,    ## Train  Validation Test split
          'max_seq_len': 400,   # Length of padding and input of Embedding layer
          'promath_dir': 'promath', # name of dir with the processed arXiv tar files
          #'save_path': 'glossary/test_lstm', #Path to save the positive results
          'min_words': 15, # min number of words for paragraphs to be considered
          'model_type': 'lstm',  # options are lstm or conv
          'profiling': False,     # T/F whether to add the callback for profiling
          }

    cfg['base_dir'] = os.environ.get('PERMSTORAGE', '/media/hd1') # This is permanent storage
    cfg['local_dir'] = os.environ.get('TEMPFASTSTORAGE', '/tmp')  # This is temporary fast storage
    #xml_lst += glob('/media/hd1/training_defs/math14/*.xml.gz')


    # CREATE LOG FILE AND OBJECT
    hoy = dt.now()
    timestamp = hoy.strftime("%b-%d_%H-%M")
    if cfg['model_type'] == 'lstm':
        save_path_dir = os.path.join(cfg['local_dir'], 'trained_models/lstm_classifier/lstm_' + timestamp)
    else:
        save_path_dir = os.path.join(cfg['local_dir'], 'trained_models/conv_classifier/conv_' + timestamp)

    os.makedirs(save_path_dir)
    cfg['save_path'] = os.path.join(save_path_dir, 'classification_results')


    if len(args) > 0:
        cfg['profiling'] = args[0].profiling

    # CREATE PROFILING LOGS DIRECTORY
    if cfg['profiling'] == True:
        cfg['prof_dir'] = os.path.join(save_path_dir, 'profiling_log')
        os.makedirs(cfg['prof_dir'], exist_ok=True)
    return save_path_dir, cfg

def model_callbacks(cfg):
    '''
    Return a Tensorboard Callback
    '''
    if cfg['profiling'] == True:
        return [TensorBoard(log_dir=cfg['prof_dir'],
                                histogram_freq=1,
                                profile_batch="2,22")]
    else:
        return []

# ++++++++++ Accesory Functions +++++++++++++++++++
def text2seq(text,tkn2idx):
    if type(text) == str:
        text = normalize_text(text).split()
    return [tkn2idx.get(s, 0) for s in text]


def padding_fun(seq, tkn2idx_value, cfg):
    # Apply pad_sequence using the cfg dictionary
    return pad_sequences(seq, maxlen=cfg['max_seq_len'],
                            padding='post', 
                            truncating='post',
                            value=tkn2idx_value) 

# READ THE TRAINING DATA
def read_train_data(xml_lst, cfg):
    stream = stream_arxiv_paragraphs(xml_lst, samples=cfg['batch_size'])

    all_data = []
    for s in stream:
        try:
            all_data += list(zip(s[0], s[1]))
        except IndexError:
            logger.warning('Index error in the data stream.')
    shuffle(all_data)

    # Split the ranges first because all_data is a big object
    r_all_data = range(len(all_data))
    TVT_len = int(cfg['TVT_split']*len(all_data))
    r_training =  r_all_data[:TVT_len] 
    r_validation = r_all_data[TVT_len:] 
    # split again half of validation for test
    r_test = r_validation[:int(0.5*len(r_validation))]
    r_validation = r_validation[int(0.5*len(r_validation)):]
    log_str = 'Original Range: {}\n Training: {}  Validation: {}  Test: {} \n'\
                  .format(repr(r_all_data), repr(r_training), repr(r_test), repr(r_validation))  

    logger.info(log_str)


    #Split the data and convert into test[0]: tuple of texts
    #                                test[1]: tuple of labels
    training = list(zip(*( [all_data[k] for k in r_training] )))
    validation = list(zip(*( [all_data[k] for k in r_validation] )))
    test = list(zip(*( [all_data[k] for k in r_test] )))

    tknr = Counter()
    # Normally test data is not in the tokenization
    # but this is text mining not statistical ML
    for t in all_data:
        tknr.update(normalize_text(t[0]).split())
    logger.info("Most common tokens are: {}".format(tknr.most_common()[:10]))

    idx2tkn = list(tknr.keys())
    # append a padding value
    idx2tkn.append('�')
    tkn2idx = {tok: idx for idx, tok in enumerate(idx2tkn)}
    word_example = 'commutative'
    idx_example = tkn2idx[word_example]
    cfg['tot_words'] = len(idx2tkn)
    logger.info('Index of "{0}" is: {1}'.format(word_example, idx_example ))
    logger.info(f"idx2tkn[{idx_example}] = {idx2tkn[idx_example]}")
    logger.info('index of padding value is: {}'.format(tkn2idx['�']))


    # CREATE THE DATA SEQUENCES
    train_seq = [text2seq(t, tkn2idx) for t in training[0]]
    validation_seq = [text2seq(t, tkn2idx) for t in validation[0]]
    test_seq = [text2seq(t, tkn2idx) for t in test[0]]

    # PAD THE SEQUENCES AND KEEP THE VARIABLE NAME
    tkn2idx_value = tkn2idx['�']
    train_seq = padding_fun(train_seq, tkn2idx_value, cfg)
    validation_seq = padding_fun(validation_seq, tkn2idx_value, cfg)
    test_seq = padding_fun(test_seq, tkn2idx_value, cfg)

    return train_seq, validation_seq, test_seq, idx2tkn, tkn2idx, training, validation, test, cfg

def gen_embed_matrix(tkn2idx, cfg):
    coverage_cnt = 0
    cfg['wembed_path'] = os.path.join(cfg['base_dir'], 'embeddings/model14-14_12-08/vectors.bin')
    with open_w2v(cfg['wembed_path']) as embed_dict:
        cfg['embed_dim'] = embed_dict[next(iter(embed_dict))].shape[0]
        embed_matrix = np.zeros((cfg['tot_words'], cfg['embed_dim']))
        for word, ind in tkn2idx.items():
            vect = embed_dict.get(word)
            if vect is not None:
                #vect = vect/np.linalg.norm(vect)
                embed_matrix[ind] = vect
                coverage_cnt += 1

    logger.info("The coverage percentage is: {0:2.1f}".format(coverage_cnt/len(tkn2idx)))

    return embed_matrix, cfg

def conv_model_globavgpool(embed_matrix, cfg):
    conv_m = Sequential([
        Embedding(cfg['tot_words'], cfg['embed_dim'],
                  input_length=cfg['max_seq_len'], weights=[embed_matrix], trainable=False),
        Conv1D(cfg['conv_filters'], cfg['kernel_size'], activation='relu'),
        GlobalAveragePooling1D(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    conv_m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    conv_m.summary(print_fn=logger.info) 
    return conv_m

def lstm_model_one_layer(embed_matrix, cfg):
    lstm_model = Sequential([
        Embedding(cfg['tot_words'], cfg['embed_dim'], 
                  input_length=cfg['max_seq_len'],
                  weights=[embed_matrix],
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

# Find the best classification cutoff parameter
def find_best_cutoff(model, val_seq, validation):
    '''
    model: is an instance with predict attribute
    val_seq: has same shape and format as the training sequence
    '''
    f1_max = 0.0; opt_prob = None
    pred_validation = model.predict(val_seq)

    for thresh in np.arange(0.1, 0.901, 0.01):
        thresh = np.round(thresh, 2)
        f1 = metrics.f1_score(validation[1], (pred_validation > thresh).astype(int))
        #print('F1 score at threshold {} is {}'.format(thresh, f1))
        
        if f1 > f1_max:
            f1_max = f1
            opt_prob = thresh
    return (opt_prob, f1_max)


def argParse():
    '''
    Parsing all arguments
    '''
    import argparse
    parser = argparse.ArgumentParser(description="Train LSTM model")
    parser.add_argument('--epochs', type=int, default=2,
            help="Number of epochs to train. Overrides default value")
    parser.add_argument('-p', '--profiling', action='store_true',
            help="Set the profiling mode to True (default False)")
    args = parser.parse_args()
    return args

##### Parameters of Features to explore
# length of padding (number of input tokens)
# length of lstm cells
# change the word embedding model, this should a list of available word embedding with a method to open them

def main():
    args = argParse()
    save_path_dir, cfg = gen_cfg(args)
    xml_lst = glob(cfg['base_dir'] + cfg['glob_data_source'])

    logging.basicConfig(filename=os.path.join(save_path_dir, 'training.log'),
            level=logging.INFO)
    logger.info("GPU devices: {}".format(list_physical_devices('GPU')))
    logger.info("Length of the xml_lst is: {}".format(len(xml_lst)))

    train_seq, validation_seq, test_seq, idx2tkn, tkn2idx, training, validation, test, cfg = read_train_data(xml_lst, cfg)

    embed_matrix, cfg = gen_embed_matrix(tkn2idx, cfg)

    if cfg['model_type'] == 'lstm':
        cfg['lstm_cells'] = 128 # Required LSTM layer parameter
        cfg['epochs'] = args.epochs 
        cfg['model_name'] = lstm_model_one_layer.__name__

        model = lstm_model_one_layer(embed_matrix, cfg)
    elif cfg['model_type'] == 'conv':
        cfg['conv_filters'] = 1024 # 256
        cfg['kernel_size'] = 20 # 10
        cfg['epochs'] = 20 # 35 
        cfg['model_name'] = conv_model_globavgpool.__name__

        model = conv_model_globavgpool(embed_matrix, cfg)
    else:
        raise NotImplementedError(f'Model Type: {cfg["model_type"]} not defined')

    ### FIT THE MODEL ###
    history = model.fit(train_seq, np.array(training[1]),
                    epochs=cfg['epochs'], validation_data=(validation_seq, np.array(validation[1])),
                    batch_size=512,
                    verbose=1,
                    callbacks=model_callbacks(cfg))
    opt_prob, f1_max = find_best_cutoff(model, validation_seq, validation)

    logger.info('\n Optimal probabilty threshold is {} for maximum F1 score {}\n'\
            .format(opt_prob, f1_max))

    cfg['opt_prob'] = opt_prob

    pred_test = model.predict(test_seq)

    metrics_str = metrics.classification_report((pred_test > opt_prob).astype(int), test[1])
    logger.info('\n' + metrics_str)


    model.save_weights( os.path.join(save_path_dir, 'model_weights') )

    # Log both a pretty printed and a copy-pasteable version of the the cfg
    # dictionary
    logger.info('\n'.join(["{}: {}".format(k,v) for k, v in cfg.items()]))
    logger.info(repr(cfg))


    with open(os.path.join(save_path_dir, 'cfg_dict.json'), 'w') as cfg_fobj:
        json.dump(cfg, cfg_fobj)

    with open(os.path.join(save_path_dir,'idx2tkn.pickle'), 'wb') as idx2tkn_fobj:
        pickle.dump(idx2tkn, idx2tkn_fobj, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
