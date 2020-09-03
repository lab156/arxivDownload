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

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional,\
                      GRU, Dropout, GlobalAveragePooling1D, Conv1D
from tensorflow.keras.models import Sequential

import sklearn.metrics as metrics
import matplotlib.pyplot as plt
# -

# %load_ext autoreload
# %autoreload 2
from embed_utils import open_w2v
import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from classifier_trainer.trainer import stream_arxiv_paragraphs

tf.pla

# +
cfg = {'batch_size': 5000}
xml_lst = glob('/media/hd1/training_defs/math15/*.xml.gz')
xml_lst += glob('/media/hd1/training_defs/math14/*.xml.gz')
stream = stream_arxiv_paragraphs(xml_lst, samples=cfg['batch_size'])

validation = next(stream)
training = [[], []]
for s in stream:
    training[0] += s[0]
    training[1] += s[1]
# -

tknr = Tokenizer()
tknr.fit_on_texts(list(validation[0]) + training[0])
tot_words = len(tknr.word_index) + 1
train_seq = tknr.texts_to_sequences(training[0])
validation_seq = tknr.texts_to_sequences(validation[0])

embed_matrix = np.zeros((tot_words, 200))
coverage_cnt = 0
with open_w2v('/media/hd1/embeddings/model14-14_12-08/vectors.bin') as embed_dict:
    for word, ind in tknr.word_index.items():
        vect = embed_dict.get(word)
        if vect is not None:
            vect = vect/np.linalg.norm(vect)
            embed_matrix[ind] = vect
            coverage_cnt += 1

max_seq_len = 400
train_seq = pad_sequences(train_seq, maxlen=max_seq_len, padding='pre')
validation_seq = pad_sequences(validation_seq, maxlen=max_seq_len, padding='pre')

conv_model = Sequential([
    Embedding(tot_words, 200, input_length=max_seq_len, weights=[embed_matrix], trainable=False),
    Conv1D(128, 5, activation='relu'),
    GlobalAveragePooling1D(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid'),
])
conv_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
conv_model.summary()
history = conv_model.fit(train_seq, np.array(training[1]),
                epochs=20, validation_data=(validation_seq, np.array(validation[1])),
                batch_size=512,
                verbose=1)

# Conv stats results
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
predictions = conv_model.predict(validation_seq)
print(metrics.classification_report(np.round(predictions), validation[1]))

lstm_model = Sequential([
    Embedding(tot_words, 200, input_length=max_seq_len, weights=[embed_matrix], trainable=False),
    Bidirectional(LSTM(128, return_sequences=True)),
    GlobalAveragePooling1D(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
class_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
class_model.summary()
history = class_model.fit(train_seq, np.array(training[1]),
                epochs=7, validation_data=(validation_seq, np.array(validation[1])),
                batch_size=512,
                verbose=1)


# +
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
    
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
# -

predictions = class_model.predict(validation_seq)
print(metrics.classification_report(np.round(predictions), validation[1]))
