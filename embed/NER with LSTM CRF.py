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
                      GRU, Dropout, GlobalAveragePooling1D, Conv1D, TimeDistributed
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import re
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
import nltk.data
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
import pickle
import math
#import collections.Iterable as Iterable

import sklearn.metrics as metrics
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2
import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from unwiki import unwiki
import ner
from embed_utils import open_w2v

# +
with open('/media/hd1/wikipedia/wiki_definitions_improved.txt', 'r') as wiki_f:
    wiki = wiki_f.readlines()
    
# Get data and train the Sentence tokenizer
# Uses a standard algorithm (Kiss-Strunk) for unsupervised sentence boundary detection
text = ''
for i in range(1550):
    text += unwiki.loads(eval(wiki[i].split('-#-%-')[2]))

trainer = PunktTrainer()
trainer.INCLUDE_ALL_COLLOCS = True
trainer.train(text)
sent_tok = PunktSentenceTokenizer(trainer.get_params())
print(sent_tok._params.abbrev_types)
# -

# ### TODO
# * protect _inline_math_ from keras tokenizer, right now it is breaking it up
# * Search for a minimal stemmer that finds for example zero-sum games in zero-sum game or absolute continuity 

# +
cfg = {}

word_tok = Tokenizer(oov_token='<UNK>')
clean_str = lambda s: unwiki.loads(eval(s)).replace('\n', ' ')
fields = {'texts': [], 'titles': [], }
for w in wiki:
    title, section, defin_parag = w.split('-#-%-')
    defin_parag = clean_str(defin_parag)
    for defin in sent_tok.tokenize(defin_parag):
        fields['titles'].append(title.lower().strip())
        fields['texts'].append(defin)
word_tok.fit_on_texts(fields['titles'] + fields['texts'])

rev_word_index = (1 + len(word_tok.word_index))*['***']
for word,ind in word_tok.word_index.items():
    rev_word_index[ind] = word
# -

fields['texts'][1683]

fields['labels'] = []
fields['tokens'] = word_tok.texts_to_sequences(fields['texts'])
empty_sentence_lst = []
for N in range(len(fields['texts'])):
    title_lst = word_tok.texts_to_sequences([fields['titles'][N].strip()])[0]
    tags = ner.bio_tag.bio_tkn_tagger(title_lst, fields['tokens'][N] )
    try:
        fields['labels'].append(list(zip(*tags))[1])
    except IndexError:
        fields['labels'].append(['0'])
        empty_sentence_lst.append(N)
print(f'Found {len(empty_sentence_lst)} empty sentences')

K = 31765
Tex = fields['texts'][K]
Tok = fields['tokens'][K]
Lab = fields['labels'][K]
Tit = fields['titles'][K]
print(f'the title of the article is: {Tit}')
for ind, t in enumerate(Tok):
    print('{0:>5} {1:>12} {2:>5}'.format(t, rev_word_index[t], Lab[ind]))

cfg['maxlen'] = max([len(l) for l in fields['tokens']])//12
cfg['padding'] = 'pre'
train_seq = pad_sequences(fields['tokens'], maxlen=cfg['maxlen'], padding=cfg['padding'])
train_lab = pad_sequences(fields['labels'], maxlen=cfg['maxlen'], padding=cfg['padding'])
train_seq2 = []
train_lab2 = []
for ind, t in enumerate(train_lab):
    if 2 in t:
        train_seq2.append(train_seq[ind])
        train_lab2.append(train_lab[ind])
train_seq2 = np.array(train_seq2)
train_lab2 = np.array(train_lab2)


# +
class NerModel(tf.keras.Model):
    def __init__(self, hidden_num, vocab_size, label_size, embedding_size):
        super(NerModel, self).__init__()
        self.hidden_num = hidden_num
        self.vocab_size = vocab_size
        self.label_size = label_size
        
        self.embedding = Embedding(vocab_size, embedding_size)
        self.biLSTM = Bidirectional(LSTM(hidden_num, return_sequences=True))
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense = Dense(label_size)
        
        self.transition_params = tf.Variable(tf.random.uniform(shape=(label_size, label_size)))
        
    def call(self, text, labels=None, training=None):
        text_lens = tf.math.reduce_sum(tf.cast(tf.math.not_equal(text, 0), dtype=tf.int32), axis=-1)
        inputs = self.embedding(text)
        inputs = self.biLSTM(inputs)
        inputs = self.dropout(inputs, training)
        logits = self.dense(inputs)
        
        if labels is not None:
            label_sequences = tf.convert_to_tensor(labels, dtype=tf.int32)
            log_likelihood, self.transition_params = \
            tfa.text.crf_log_likelihood(logits, label_sequences, text_lens,
                                        transition_params=self.transition_params)
            return logits, text_lens, log_likelihood
        else:
            return logits, text_lens
        
#model.summary()


# +
# Train NER model
cfg['learning_rate'] = 0.1
model = NerModel(64, len(word_tok.word_index)+1, 4, 100)
optimizer = tf.keras.optimizers.Adam()
def train_one_step(text_batch, labels_batch):
    with tf.GradientTape() as tape:
        logits, text_lens, log_likelihood = model(text_batch, labels_batch, training=True)
        loss = - tf.reduce_mean(log_likelihood)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, logits, text_lens

def get_acc_one_step(logits, text_lens, labels_batch):
    paths = []
    accuracy = 0
    for logit, text_len, labels in zip(logits, text_lens, labels_batch):
        viterbi_path, _ = tfa.text.viterbi_decode(logit[:text_len], model.transition_params)
        paths.append(viterbi_path)
        correct_prediction = tf.equal(
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([viterbi_path],
                                                            padding='post'), dtype=tf.int32),
            tf.convert_to_tensor(tf.keras.preprocessing.sequence.pad_sequences([labels[:text_len]],
                                                            padding='post'), dtype=tf.int32)
        )
        accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
    accuracy = accuracy / len(paths)
    return accuracy

best_acc = 0
step = 0
epochs = 20
bs = 1000
for epoch in range(epochs):
    for (text_batch, labels_batch) in \
    [[train_seq2[bs*i:bs*(i+1)], train_lab2[bs*i:bs*(i+1)]]\
     for i in range(math.ceil(len(train_seq2)/bs))]:
        step = step + 1
        loss, logits, text_lens = train_one_step(text_batch, labels_batch)
        if step % 20 == 0:
            accuracy = get_acc_one_step(logits, text_lens, labels_batch)
            print('epoch %d, step %d, loss %.4f , accuracy %.4f' % (epoch, step, loss, accuracy))
            if accuracy > best_acc:
                best_acc = accuracy
                #ckpt_manager.save()
                print("model saved")
# -

model.summary()

sample_str = 'A banach space is defined as named entity recognition'
sample_tok = word_tok.texts_to_sequences([sample_str])
sample_pad = pad_sequences(sample_tok, maxlen=cfg['maxlen'], padding=cfg['padding'])
pred = [model.predict(text_batch[i])[1] for i in range(len(text_batch))]

# +
cf = {'input_dim': len(word_tok.word_index)+1,
      'output_dim': 25,
     'input_length': max([len(l) for l in fields['tokens']])//12,
     'n_tags': 4,
     'batch_size': 1000}
    
# Define the categorical labels
train_lab2_cat = np.array([to_categorical(c, num_classes=cf['n_tags']) for c in train_lab2])
embed_matrix = np.zeros((cf['input_dim'], 200))
coverage_cnt = 0
with open_w2v('/media/hd1/embeddings/model14-14_12-08/vectors.bin') as embed_dict:
    for word, ind in word_tok.word_index.items():
        vect = embed_dict.get(word)
        if vect is not None:
            vect = vect/np.linalg.norm(vect)
            embed_matrix[ind] = vect
            coverage_cnt += 1
# -

cf = {'input_dim': len(word_tok.word_index)+1,
      'output_dim': 25,
     'input_length': max([len(l) for l in fields['tokens']])//12,
     'n_tags': 4,
     'batch_size': 1000}


# +
# DEFINE MODEL WITH biLSTM AND TRAIN FUNCTION    
def get_bilstm_lstm_model(cfg_dict):
    model = Sequential()
    # Add Embedding layer
   # model.add(Embedding(cfg_dict['input_dim'], 
   #                     output_dim=cfg_dict['output_dim'],
   #                     input_length=cfg_dict['input_length'],
   #                    weights = [embed_matrix],
   #                    trainable = False))
    model.add(Embedding(cfg_dict['input_dim'], 
                        output_dim=cfg_dict['output_dim'],
                        input_length=cfg_dict['input_length']))
    # Add bidirectional LSTM
    model.add(Bidirectional(LSTM(units=cfg_dict['output_dim'],
                                 return_sequences=True,
                                 dropout=0.2, 
                                 recurrent_dropout=0.2), merge_mode = 'concat'))
    # Add LSTM
    model.add(LSTM(units=cfg_dict['output_dim'],
                   return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                   recurrent_initializer='glorot_uniform'))
    # Add timeDistributed Layer
    model.add(TimeDistributed(Dense(cfg_dict['n_tags'], activation="relu")))
    #Optimiser 
    adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    
    return model

def train_model(X, y, model, epochs=10):
    out = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}
    # fit model for one epoch on this sequence
    res = model.fit(X, y, verbose=1, epochs=epochs,
                    batch_size=cf['batch_size'],
                    validation_split=0.2 )
    out['accuracy'].append(res.history['accuracy'])
    out['val_accuracy'].append(res.history['val_accuracy'])
    out['loss'].append(res.history['loss'])
    out['val_loss'].append(res.history['val_loss'])
    return out
model_bilstm_lstm = get_bilstm_lstm_model(cf)
#plot_model(model_bilstm_lstm)
# -

train_lab2_cat = np.array([to_categorical(c, num_classes=cf['n_tags']) for c in train_lab2])
history = train_model(train_seq2, train_lab2_cat, model_bilstm_lstm, epochs=250)

# + jupyter={"outputs_hidden": true}
sample_str = 'banach spaces are defined as complete vector space of some king'
sample_tok = word_tok.texts_to_sequences([sample_str])
#sample_pad = pad_sequences(sample_tok, maxlen=cf['input_length'], padding=cfg['padding'])
sample_pad = train_seq2[N]
pred = model_bilstm_lstm.predict(sample_pad)
#np.argmax(pred.squeeze(), axis=1)
pred
# -

train_seq2[N]

T = train_seq2[N].reshape(1,42)
model_bilstm_lstm.predict(T)


def decoder(T, L):
    pred = model_bilstm_lstm.predict(T.reshape(1,42))
    P = pred #np.argmax(pred.squeeze(), axis=1)
    for ind, t in enumerate(T):
        if t != 0:
            print("{0:>22}: {1:} {2:}".format(rev_word_index[t], L[ind], P[ind]))
#decoder(sample_pad[0], np.argmax(pred.squeeze(), axis=1))
N = -27
decoder(train_seq2[N], train_lab2[N])


# +
def plot_graphs(history, string, start_at=0):
    plt.plot(history[string][start_at:])
    plt.plot(history['val_'+string][start_at:])
    plt.xlabel('Epochs')
    plt.ylabel(string[start_at:])
    plt.legend([string, 'val_'+string])
    plt.show()
    
plot_graphs(history, "accuracy", start_at=400)
plot_graphs(history, "loss", start_at=400)
# -


