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

# + tags=["NER", "Tensorflow2.0"]
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional,\
                      GRU, Dropout,GlobalAveragePooling1D, Conv1D, TimeDistributed,\
                      Input, Concatenate, GlobalMaxPooling1D
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam
from tqdm.keras import TqdmCallback
import tensorflow as tf
import tensorflow_addons as tfa
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
import gzip
import random
#import collections.Iterable as Iterable

import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from nltk.chunk import conlltags2tree, tree2conlltags

import gzip
from lxml import etree
from tqdm import tqdm
import random
import glob
from collections import Counter

# %load_ext autoreload
# %autoreload 2
import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from unwiki import unwiki
import ner
from embed_utils import open_w2v
import clean_and_token_text as clean
import train_ner as TN
import inference_ner as IN

# +
# PREPARE THE DATA  
cfg = TN.gen_cfg()

text_lst = TN.get_wiki_pm_stacks_data(cfg)
sent_tok, tok_params = TN.gen_sent_tokzer(text_lst, cfg)
#logger.info(sent_tok._params.abbrev_types)

def_lst = ner.bio_tag.put_pos_ner_tags(text_lst, sent_tok)
random.shuffle(def_lst)

pos_ind_dict, pos_lst, cfg = TN.get_pos_ind_dict(def_lst, cfg)

wind, embed_matrix, cfg = TN.open_word_embedding(cfg)

train_def_lst, test_def_lst, valid_def_lst = TN.get_ranges_lst(def_lst, cfg)

train_seq, train_pos_seq, train_bin_seq , train_lab = TN.prep_data4real(
        train_def_lst, wind, pos_ind_dict, cfg)
test_seq, test_pos_seq, test_bin_seq , test_lab = TN.prep_data4real(
        test_def_lst, wind, pos_ind_dict, cfg)
valid_seq, valid_pos_seq, valid_bin_seq , valid_lab = TN.prep_data4real(
        valid_def_lst, wind, pos_ind_dict, cfg)


# + magic_args="echo dont want to join test and validation" language="script"
# # JOIN TEST AND VALIDATION
# print(train_seq.shape)
# print(test_seq.shape)
# print(valid_seq.shape)
# testq_seq = np.concatenate((test_seq, valid_seq), axis=0)
# testq_lab = np.concatenate((test_lab, valid_lab), axis=0)
# testq_pos_seq = np.concatenate((test_pos_seq, valid_pos_seq), axis=0)
# testq_bin_seq = np.concatenate((test_bin_seq, valid_bin_seq), axis=0)
# testq_bin_seq.shape

# +
cfg.update({'input_dim': len(wind),
      'output_dim': 200, #don't keep hardcoding this
     'input_length': cfg['padseq']['maxlen'],
     'pos_dim': 3,
            'pos_constraint': 1/200,
     'n_tags': 2,
     'batch_size': 2000,
     'lstm_units1': 200,
     'lstm_units2': 150,
     'adam': {'lr': 0.025, 'beta_1': 0.9, 'beta_2': 0.999},
     'epochs': 35,
           'train_test_split': 0.7})

model_bilstm = TN.bilstm_model_w_pos(embed_matrix, cfg)
#history = train_model(train_seq, train_lab, test_seq, test_lab, model_bilstm_lstm, cfg )
history = model_bilstm.fit([train_seq, train_pos_seq, train_bin_seq], 
            train_lab, epochs=cfg['epochs'], batch_size=cfg['batch_size'],
            validation_data=([test_seq, test_pos_seq, test_bin_seq], test_lab))
# -


with open_w2v('/media/hd1/embeddings/model4ner_19-33_02-01/vectors.bin') as embed_dict:
    wind = ['<UNK>',] + list(embed_dict.keys())
    cfg['emb_nvocab'] = len(wind) 
    embed_matrix = np.zeros((cfg['emb_nvocab'], 200))
    for word, vec in embed_dict.items():
        #vect = embed_dict.get(word)
        ind = wind.index(word)
            #vect = vect/np.linalg.norm(vect)
        embed_matrix[ind] = vec
#print("Coverage of embed is: {}".format(coverage_cnt/len(embed_dict)))

sent_lengths = []
for d in def_lst:
    slen = min(len(d['ner']), 150)
    sent_lengths.append(slen)
plt.hist(sent_lengths, bins=15)
plt.title('length of selected sentences with definiendum')
plt.show()

# + jupyter={"source_hidden": true}
cfg['padseq'] = {'maxlen': 50 , 'padding': 'post', 'truncating': 'post'}
cfg['n_tags'] = 2

# shuffle def_lst
random.shuffle(def_lst)

cfg['train_test_split'] = 0.9
r_def_lst = range(len(def_lst))
TVT_len = int(cfg['train_test_split']*len(def_lst))
r_train = r_def_lst[:TVT_len]
r_valid = r_def_lst[TVT_len:]
r_test = r_valid[:int(0.5*len(r_valid))]
r_valid = r_valid[int(0.5*len(r_valid)):]
log_str = 'Original Range: {}\n Training: {}  Validation: {}  Test: {} \n'\
              .format(repr(r_def_lst), repr(r_train), repr(r_test), repr(r_valid)) 
print(log_str)

train_def_lst = [def_lst[k] for k in r_train]
test_def_lst = [def_lst[k] for k in r_test]
valid_def_lst = [def_lst[k] for k in r_valid]

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

#binary_fun_lst = [
#    lambda w: word[0] in string.ascii_uppercase,  # Capitalized
#]
def binary_features(dat):
    out_lst = []
    for d in dat:
        word =d[0][0]
        capitalized = float(word[0] in string.ascii_uppercase)
        contains_dash = float('-' in word)
        
        out_lst.append((capitalized, contains_dash))
    return out_lst

cfg['nbin_feats'] = 2 # number of binary features defined in the function above
    
def prep_data4real(_def_lst, wind, cfg):
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

train_seq, train_pos_seq, train_bin_seq , train_lab = prep_data4real(train_def_lst, wind, cfg)
test_seq, test_pos_seq, test_bin_seq , test_lab = prep_data4real(test_def_lst, wind, cfg)
valid_seq, valid_pos_seq, valid_bin_seq , valid_lab = prep_data4real(valid_def_lst, wind, cfg)

# -

# ### TODO
# * Right the different concatenated pieces are in different orders of magnitude. Normalization might help
# * Search for a minimal stemmer that strips plural or adverbial suffices for example zero-sum games in zero-sum game or absolute continuity and absolute continuous

# + jupyter={"outputs_hidden": true}
cfg.update({'input_dim': len(wind),
      'output_dim': 200, #don't keep hardcoding this
     'input_length': cfg['padseq']['maxlen'],
     'pos_dim': 3,
            'pos_constraint': 1/200,
     'n_tags': 2,
     'batch_size': 2000,
     'lstm_units': 150,
      'adam': {'lr': 0.0005, 'beta_1': 0.9, 'beta_2': 0.999}})

model_bilstm = bilstm_model_w_pos(embed_matrix, cfg)
    #history = train_model(train_seq, train_lab, test_seq, test_lab, model_bilstm_lstm, cfg )
history = model_bilstm.fit([train_seq, train_pos_seq, train_bin_seq],
       train_lab, epochs=40,
       batch_size=cfg['batch_size'],
       validation_data=([test_seq, test_pos_seq, test_bin_seq], test_lab))

# + magic_args="echo no need for loading right now" language="script"
# #res = with_pos.fit([train_seq, train_pos_seq, train_bin_seq], train_lab, verbose=1, epochs=30,
# #                batch_size=cfg['batch_size'],
# #                validation_data=([test_seq, test_pos_seq, test_bin_seq], test_lab))
#
# # Load weights instead of training
# save_model_dir = '/home/luis/ner_model/'
# with open(save_model_dir + 'cfg.json', 'r') as cfg_fobj:
#     cfg = json.load(cfg_fobj)
#     
# with_pos = bilstm_lstm_model_w_pos(cfg)
# res = with_pos.load_weights(save_model_dir + 'bilstm_with_pos')

# +
# #%%script echo no train loading 
#history = train_model(train_seq, train_lab, model_bilstm_lstm, epochs=70)

#r = history
r = history
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(121)
ax1.plot(r.history['loss'], label='loss')
ax1.plot(r.history['val_loss'], label='val_loss')
ax1.grid()
ax1.legend()
ax2 = plt.subplot(122)
ax2.plot(r.history['accuracy'], label='acc')
ax2.plot(r.history['val_accuracy'], label='val_acc')
ax2.grid()
ax2.legend()

# + magic_args="echo skip this" language="script"
# #sample_str = 'a banach space is defined as complete vector space of some kind .'
# #sample_str = 'We define a shushu space as a complete vector space of some kind .'
# sample_str = '_display_math_ The Ursell functions of a single random variable X are obtained from these by setting _inline_math_..._inline_math_ .'
# sample_pad, _ = prep_data(sample_str, wind, cfg, 'no_tags')
# sample_pad = pad_sequences([sample_pad], **cfg['padseq'])
# print(sample_pad)
# pred = model_bilstm.predict(sample_pad)
# np.argmax(pred.squeeze(), axis=1)
# for i, w in enumerate(sample_pad[0]):
#     if wind[w] == '.':
#         break
#     print(wind[w], np.round(pred)[0][i])
#     if wind[w] == '.':
#         break


# +
#preds = model_bilstm_lstm.predict(test_seq)
preds = model_bilstm.predict([test_seq, test_pos_seq, test_bin_seq])

l,a = model_bilstm.evaluate([test_seq, test_pos_seq, test_bin_seq], test_lab)
# -

a

k = 90
for i in range(len(preds[k])):
    try:
        print('{:<20} {} {:1.2f}'.format(test_def_lst[k]['ner'][i][0][0], 
                                         test_def_lst[k]['ner'][i][1],
                                         round(preds[k][i][0],2)))
    except IndexError:
        break

# +
# use the TN.get_chunkscore function
data_points = []
BOY_f_score = (0, 0) # (CutOff, score)
for co in np.arange(0.1, 1, 0.05):
    cs = get_chunkscore(co)
    data_points.append((cs.accuracy(), cs.f_measure()))
    if cs.f_measure() > BOY_f_score[1]:
        BOY_f_score = (co, cs.f_measure())

plt.plot(list(zip(*data_points))[0], label='acc')
plt.plot(list(zip(*data_points))[1], label='F1')
plt.legend()
plt.show()

print(get_chunkscore(BOY_f_score[0]))
print(f"Cutoff:  {BOY_f_score[0]}")
cfg['tboy'] = {'cutoff': BOY_f_score[0], 
              'f_meas': BOY_f_score[1]}
# -

tboy_finder(model_bilstm, test_seq, 
            test_pos_seq, test_bin_seq, 
            test_lab,test_def_lst,cfg)

# +
#1/5404.0*(np.sum(test_lab*np.log(np.squeeze(preds))) + np.sum((1-test_lab)*np.log(np.squeeze(1-preds))))

# + magic_args="echo Dumping files careful" language="script"
# # SAVE AND LOAD MODELS WEIGHTS
# cfg['save_model_dir'] = '/home/luis/ner_model/'
# #with_pos.save_weights(cfg['save_model_dir'] + 'bilstm_with_pos')
# with open(cfg['save_model_dir'] + 'cfg.json', 'w') as cfg_fobj:
#     json.dump(cfg, cfg_fobj)
# with open(os.path.join(cfg['save_model_dir'],'wordindex.pickle'), 'wb') as wind_fobj:
#     pickle.dump(wind, wind_fobj) 
# with open(os.path.join(cfg['save_model_dir'],'posindex.pickle'), 'wb') as pos_fobj:
#     pickle.dump(pos_lst, pos_fobj)
#     
# with open('/home/luis/ner_model/punkt_params.pickle', 'wb') as punkt_fobj:
#     pickle.dump(trainer.get_params(), punkt_fobj)

# +
base_path = '/tmp/rm_me_NER/trained_ner/lstm_ner/ner_Sep-26_15-41/'
cfg = IN.open_cfg_dict(base_path + 'cfg.json')

wind, tkn2idx = IN.read_word_index_tkn2idx(os.path.join(base_path,\
            'wordindex.pickle'))

pos_lst, pos_ind_dict = IN.read_pos_index_pos2idx(os.path.join(base_path,\
            'posindex.pickle'))

sent_tok = IN.read_sent_tok(os.path.join(base_path, 'punkt_params.pickle'))

bi_model = tf.keras.models.load_model(os.path.join(base_path, 'bilstm_with_pos'))
# -

tkn2idx = {tok: idx for idx, tok in enumerate(wind)}
#pos_dict = {tok: idx for idx, tok in enumerate(pos_lst)}
pos_dict = pos_ind_dict

loss, acc = bi_model.evaluate([train_seq, train_pos_seq, train_bin_seq], train_lab)

# +
#text = D.find('stmnt').text
#text = '_display_math_ The Toledo functions of a single random variable X are obtained from these by setting _inline_math_..._inline_math_ .'
text = 'We call cookie cutter the functions from nowhere to somewhere.'
print(text)
ww, pp, bb = IN._prep_raw_data(text, sent_tok, tkn2idx, pos_dict, cfg)
preds = bi_model.predict([ww,pp, bb])

for k in range(preds.shape[0]):
    sent_str = ' '.join(["{:1.2f}".format(r[0]) for r in preds[k]])
    print(sent_str)

# -

with gzip.open('/home/luis/rm_me2.tar.gz', 'wb') as fobj:
    fobj.write(etree.tostring(root, pretty_print=True))

# # get_bilstm_lstm_model Training history
# ## First working attempt:  commit e4c41f0
#
# Epochs: 70 [01:00<00:00, 3.00s/epoch, loss=0.0513, accuracy=0.98, val_loss=0.0636, val_accuracy=0.975]
#
# * Same attempt but first ChunkScore Epochs: approx 80,  commit: 8a3678c
#         ChunkParse score:
#             IOB Accuracy:  86.3%%
#             Precision:     56.4%%
#             Recall:        53.8%%
#             F-Measure:     55.1%%
#             
# ## Working attempt with POS: Epochs: 70
#     ChunkParse score:
#         IOB Accuracy:  85.4%%
#         Precision:     53.2%%
#         Recall:        47.6%%
#         F-Measure:     50.3%%
# ### Added both LSTMs Bidirectional:
# In the previous models the second LSTM layer was not Bidirectional. This makes no sense
# loss: 0.0446 - accuracy: 0.9824 - val_loss: 0.0601 - val_accuracy: 0.9770 commit: c53ab68
#
#     ChunkParse score:
#         IOB Accuracy:  85.1%%
#         Precision:     48.7%%
#         Recall:        62.8%%
#         F-Measure:     54.9%%
#         
# With just 150 lstm units. Commit: 53dd596
#
#     ChunkParse score:
#         IOB Accuracy:  87.7%%
#         Precision:     61.4%%
#         Recall:        62.3%%
#         F-Measure:     61.8%%
#     Cutoff:  0.4
#     
# Two time dependent dense layers at the end: Commit: 2759216
#
#     ChunkParse score:
#         IOB Accuracy:  87.5%%
#         Precision:     59.5%%
#         Recall:        60.4%%
#         F-Measure:     60.0%%
#     Cutoff:  0.4
#     
# ## With Binary features: capitalize, has_dash: Commit: 09c6fcb
# loss: 0.0408 - accuracy: 0.9840 - val_loss: 0.0547 - val_accuracy: 0.9789
#
#     ChunkParse score:
#         IOB Accuracy:  87.1%%
#         Precision:     59.2%%
#         Recall:        57.9%%
#         F-Measure:     58.5%%
#     Cutoff:  0.5
#     
# ## Change initializer of POS embedding to Normal mean 0 std 1
#
# TBOY  loss: 0.0299 - accuracy: 0.9884 - val_loss: 0.0539 - val_accuracy: 0.9812
#
#     ChunkParse score:
#         IOB Accuracy:  87.3%%
#         Precision:     62.5%%
#         Recall:        64.2%%
#         F-Measure:     63.3%%
#     Cutoff:  0.4
#     
# loss: 0.0201 - accuracy: 0.9922 - val_loss: 0.0531 - val_accuracy: 0.9824 Commit: b3a1c88
#
#     ChunkParse score:
#     IOB Accuracy:  87.5%%
#     Precision:     68.9%%
#     Recall:        64.8%%
#     F-Measure:     66.8%%
#     Cutoff:  0.6
#
# loss: 0.0362 - accuracy: 0.9855 - val_loss: 0.0489 - val_accuracy: 0.9812 -- 40 epochs. Commit: 0530c59
#
#     ChunkParse score:
#         IOB Accuracy:  88.3%%
#         Precision:     62.2%%
#         Recall:        64.2%%
#         F-Measure:     63.2%%
#     Cutoff:  0.4

# + magic_args="echo Skip this" language="script"
# # This is old code -- Never worked
# # Train NER model
# cfg['learning_rate'] = 0.1
# model = NerModel(64, len(word_tok.word_index)+1, 4, 100)
# optimizer = tf.keras.optimizers.Adam()
# def train_one_step(text_batch, labels_batch):
#     with tf.GradientTape() as tape:
#         logits, text_lens, log_likelihood = model(text_batch, labels_batch, training=True)
#         loss = - tf.reduce_mean(log_likelihood)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss, logits, text_lens
#
# def get_acc_one_step(logits, text_lens, labels_batch):
#     paths = []
#     accuracy = 0
#     for logit, text_len, labels in zip(logits, text_lens, labels_batch):
#         viterbi_path, _ = tfa.text.viterbi_decode(logit[:text_len], model.transition_params)
#         paths.append(viterbi_path)
#         correct_prediction = tf.equal(
#             tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path],
#                                                             padding='post'), dtype=tf.int32),
#             tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:text_len]],
#                                                             padding='post'), dtype=tf.int32)
#         )
#         accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#         # print(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
#     accuracy = accuracy / len(paths)
#     return accuracy
#
# best_acc = 0
# step = 0
# epochs = 20
# bs = 1000
# for epoch in range(epochs):
#     for (text_batch, labels_batch) in \
#     [[train_seq2[bs*i:bs*(i+1)], train_lab2[bs*i:bs*(i+1)]]\
#      for i in range(math.ceil(len(train_seq2)/bs))]:
#         step = step + 1
#         loss, logits, text_lens = train_one_step(text_batch, labels_batch)
#         if step % 20 == 0:
#             accuracy = get_acc_one_step(logits, text_lens, labels_batch)
#             print('epoch %d, step %d, loss %.4f , accuracy %.4f' % (epoch, step, loss, accuracy))
#             if accuracy > best_acc:
#                 best_acc = accuracy
#                 #ckpt_manager.save()
#                 print("model saved")
