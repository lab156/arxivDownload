# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from glob import glob
import os
import numpy as np
from lxml import etree
from collections import Counter
from random import shuffle
import gzip

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional,\
                      GRU, Dropout, GlobalAveragePooling1D, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow_addons.callbacks import TQDMProgressBar
import tensorflow.keras.metrics as kmetrics

import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import json
import pickle
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
import peep_tar as peep
import parsing_xml as px
from extract import Definiendum

tar_tree = etree.parse('/media/hd1/training_defs/math10/1009_004.xml.gz')
tar_tree2 = etree.parse('/media/hd1/training_defs/math10/1010_001.xml.gz')
def_lst = tar_tree.findall('.//definition') + tar_tree2.findall('.//definition') 
nondef_lst = tar_tree.findall('.//nondef') + tar_tree2.findall('.//nondef')
len(def_lst) + len(nondef_lst)

# +
cfg = {'batch_size': 5000,
      'testing_size': 15000,
      'embed_dim': 200}
xml_lst = glob('/media/hd1/training_defs/math15/*.xml.gz')
xml_lst += glob('/media/hd1/training_defs/math14/*.xml.gz')
#xml_lst += glob('/media/hd1/training_defs/math01/*.xml.gz')
stream = stream_arxiv_paragraphs(xml_lst, samples=cfg['batch_size'])

all_data = []
for s in stream:
    all_data += list(zip(s[0], s[1]))
shuffle(all_data)


S = cfg['testing_size'] ## size of the test and validation sets
#Split the data and convert into test[0]: tuple of texts
#                                test[1]: tuple of labels
training = list(zip(*(all_data[2*S:])))
validation = list(zip(*(all_data[:S])))
test = list(zip(*(all_data[S:2*S])))

tknr = Counter()
# Normally test data is not in the tokenization
# but this is text mining not statistical ML
for t in all_data:
    tknr.update(normalize_text(t[0]).split())
print("Most common tokens are:", tknr.most_common()[:10])

idx2tkn = list(tknr.keys())
# append a padding value
idx2tkn.append('�')
tkn2idx = {tok: idx for idx, tok in enumerate(idx2tkn)}
word_example = 'commutative'
idx_example = tkn2idx[word_example]
cfg['tot_words'] = len(idx2tkn)
print('Index of "{0}" is: {1}'.format(word_example, idx_example ))
print(f"idx2tkn[{idx_example}] = {idx2tkn[idx_example]}")
print('index of padding value is:', tkn2idx['�'])


# +
def text2seq(text):
    if type(text) == str:
        text = normalize_text(text).split()
    return [tkn2idx.get(s, 0) for s in text]
train_seq = [text2seq(t) for t in training[0]]
validation_seq = [text2seq(t) for t in validation[0]]
test_seq = [text2seq(t) for t in test[0]]

max_seq_len = 400
padding_fun = lambda seq: pad_sequences(seq, maxlen=max_seq_len,
                                        padding='post', 
                                        truncating='post',
                                        value=tkn2idx['�']) 
train_seq = padding_fun(train_seq)
validation_seq = padding_fun(validation_seq)
test_seq = padding_fun(test_seq)
# -

embed_matrix = np.zeros((cfg['tot_words'], cfg['embed_dim']))
coverage_cnt = 0
with open_w2v('/media/hd1/embeddings/model14-14_12-08/vectors.bin') as embed_dict:
    for word, ind in tkn2idx.items():
        vect = embed_dict.get(word)
        if vect is not None:
            #vect = vect/np.linalg.norm(vect)
            embed_matrix[ind] = vect
            coverage_cnt += 1
print("The coverage percetage is: {}".format(coverage_cnt/len(idx2tkn)))

# #%%script echo skipping
# Lengths of the definitions to get the max_seq_len parameter
plt.figure(figsize=[9,6])
ax = plt.subplot(111)
plt.hist([min(len(s),2500) for s in training[0]], 100)
plt.grid()
plt.title('Length in characters of the definitions in the training set')
plt.show()

# #%%script echo skipping
cfg['conv_filters'] = 128
cfg['kernel_size'] = 5
def conv_model(cfg):
    return Sequential([
        Embedding(cfg['tot_words'], cfg['embed_dim'],
                  input_length=max_seq_len, weights=[embed_matrix], trainable=False),
        Conv1D(cfg['conv_filters'], cfg['kernel_size'], activation='relu'),
        GlobalAveragePooling1D(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
model = conv_model(cfg)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
tqdm_callback = TQDMProgressBar()
history = model.fit(train_seq, np.array(training[1]),
                epochs=20, validation_data=(validation_seq, np.array(validation[1])),
                batch_size=512,
                verbose=0,
                callbacks=[tqdm_callback])

# + magic_args="echo skipping" language="script"
# # Conv stats results
# plot_graphs(history, "accuracy")
# plot_graphs(history, "loss")
# predictions = conv_model.predict(validation_seq)
# print(metrics.classification_report(np.round(predictions), validation[1]))

# + magic_args="echo skipping" language="script"
# tar_tree = etree.parse('/media/hd1/training_defs/math99/9902_001.xml.gz')
# def_lst = tar_tree.findall('.//definition')
# nondef_lst = tar_tree.findall('.//nondef')
#
# def show_false_pos_negs(model, def_lst, nondef_lst, samples=15):
#     ex_def = [D.text for D in def_lst[:samples]]
#     ex_def_tok = padding_fun([text2seq(d) for d in ex_def])
#
#
#     ex_nondef = [D.text for D in nondef_lst[:samples]]
#     ex_nondef_tok = padding_fun([text2seq(d) for d in ex_nondef])
#
#
#     preds_nondef = model.predict(ex_nondef_tok)
#     preds_def = model.predict(ex_def_tok)
#     #print(f"Should be all zero: {preds_nondef}")
#     print('\n'.join("{0} -- {1:.3} -- {2:} -- {3:}"\
#                     .format(repr(k),float(preds_nondef[k]), len(ex_nondef[k]) ,ex_nondef[k])\
#                     for k in np.nonzero(preds_nondef.squeeze()>0.5)[0]))
#     print('\n')
#     #print(f"Should be all one: {preds_def}")
#     print('\n'.join("{0} -- {1:.3} -- {2:} -- {3:}"\
#                     .format(repr(k),float(preds_def[k]), len(ex_def[k]), ex_def[k])\
#                     for k in np.nonzero(preds_def.squeeze()<0.5)[0]))
#
# show_false_pos_negs(lstm_model, def_lst, nondef_lst, samples=20)

# + magic_args="echo skipping" language="script"
# ################## DEFINE LSTM MODEL #####################
# def lstm_single_layer_model(cfg):
#     return Sequential([
#     Embedding(cfg['tot_words'], cfg['embed_dim'], 
#               input_length=max_seq_len,
#               weights=[embed_matrix],
#               trainable=False),
#     Bidirectional(LSTM(128, return_sequences=True)),
#     GlobalAveragePooling1D(),
#     Dropout(0.2),
#     Dense(64, activation='relu'),
#     Dropout(0.2),
#     Dense(1, activation='sigmoid')
# ])
# def lstm_double_layer_model(cfg):
#     return Sequential([
#     Embedding(cfg['tot_words'], cfg['embed_dim'], 
#               input_length=max_seq_len,
#               weights=[embed_matrix],
#               trainable=False),
#     Bidirectional(LSTM(128, return_sequences=True)),
#     Dropout(0.2),
#     Bidirectional(LSTM(64, return_sequences=True)),
#     GlobalAveragePooling1D(),
#     Dropout(0.2),
#     Dense(64, activation='relu'),
#     Dropout(0.2),
#     Dense(1, activation='sigmoid')
# ])
#
# lstm_model = lstm_double_layer_model(cfg)
# lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', metrics.AUC()])
# lstm_model.summary()
# history = lstm_model.fit(train_seq, np.array(training[1]),
#                 epochs=5, validation_data=(validation_seq, np.array(validation[1])),
#                 batch_size=512,
#                 verbose=1)

# + magic_args="echo skipping" language="script"
# %%time
# # A little extra training in case the model seems to need it
# history = lstm_model.fit(train_seq, np.array(training[1]),
#                 epochs=2, validation_data=(validation_seq, np.array(validation[1])),
#                 batch_size=512,
#                 verbose=1)

# +
f1_max = 0.0; opt_prob = None
pred_validation = model.predict(validation_seq)
plot_points = []

for thresh in np.arange(0.1, 0.901, 0.01):
    thresh = np.round(thresh, 2)
    f1 = metrics.f1_score(validation[1], (pred_validation > thresh).astype(int))
    plot_points.append((thresh, f1))
    #print('F1 score at threshold {} is {}'.format(thresh, f1))
    
    if f1 > f1_max:
        f1_max = f1
        opt_prob = thresh

print('Optimal probabilty threshold is {} for maximum F1 score {}'.format(opt_prob, f1_max))
predictions = model.predict(test_seq)
print(metrics.classification_report(np.round(predictions), test[1]))
M = kmetrics.AUC()
M.update_state(test[1], predictions)
print("AUC ROC: {}".format(M.result().numpy()))


# +
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
    
#plot_graphs(history, "accuracy")
#plot_graphs(history, "loss")

# LSTM stats results
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# +
#from datetime import datetime as dt
#hoy = dt.now()
#timestamp = hoy.strftime("%H-%M_%b-%d")
#lstm_model.save_weights('/media/hd1/trained_models/lstm_classifier/one_layer_'+timestamp)
# -

P = list(zip(*plot_points))
plt.plot(P[0], P[1])

cfg

# + jupyter={"outputs_hidden": true}
# #%%script echo skipping
cfg['mnt_path'] = '/media/hd1/promath/'
cfg['save_path'] = '/home/luis/rm_me_glossary/test_conv/'

class Vectorizer():
    def __init__(self):
        pass
    def transform(self, L):
        return padding_fun([text2seq(d) for d in L])

def untar_clf_append(tfile, out_path, clf, vzer, thresh=0.5, min_words=15):
    '''
    Arguments:
    `tfile` tarfile with arxiv format ex. 1401_001.tar.gz
    `out_path` directory to save the xml.tar.gz file with the same name as `tfile`
    `clf` model with .predict() attribute
    `vzer` funtion that take the text of a paragraph and outputs padded np.arrays for `clf`
    '''
    
    root = etree.Element('root')
    for fname, tar_fobj in peep.tar_iter(tfile, '.xml'):
        try:
            DD = px.DefinitionsXML(tar_fobj) 
            if DD.det_language() in ['en', None]:
                art_tree = Definiendum(DD, model, None, vzer, None, fname=fname, thresh=opt_prob).root
                if art_tree is not None: root.append(art_tree)
        except ValueError as ee:
            print(f"{repr(ee)}, 'file: ', {fname}, ' is empty'")
    return root
    
#for k, dirname in enumerate(['tests',]):
for k, dirname in enumerate(['math01',]):
    try:
        full_path = os.path.join(cfg['mnt_path'], dirname)
        tar_lst = [os.path.join(full_path, p) for p in  os.listdir(full_path) if '.tar.gz' in p]
    except FileNotFoundError:
        print(' %s Not Found'%d)
        break
    out_path = os.path.join(cfg['save_path'], dirname)
    os.makedirs(out_path, exist_ok=True)
   
    for tfile in tar_lst:
        clf = model
        vzer = Vectorizer()
        def_root = untar_clf_append(tfile, out_path, model, vzer, thresh=opt_prob)
        #print(etree.tostring(def_root, pretty_print=True).decode())
        gz_filename = os.path.basename(tfile).split('.')[0] + '.xml.gz' 
        print(gz_filename)
        gz_out_path = os.path.join(out_path, gz_filename) 
        with gzip.open(gz_out_path, 'wb') as out_f:
            print("Writing to dfdum zipped file to: %s"%gz_out_path)
            out_f.write(etree.tostring(def_root, pretty_print=True))
# -
227/0.69


