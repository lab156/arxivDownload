from glob import glob
import os
import numpy as np
from lxml import etree
from collections import Counter
from random import shuffle
from datetime import datetime as dt

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional,\
                      GRU, Dropout, GlobalAveragePooling1D, Conv1D
from tensorflow.keras.models import Sequential


import sklearn.metrics as metrics
# -

# %load_ext autoreload
# %autoreload 2
from embed_utils import open_w2v, normalize_text
import sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from classifier_trainer.trainer import stream_arxiv_paragraphs

# +
base_dir = '/media/hd1'
cfg = {'batch_size': 5000}
xml_lst = glob(base_dir + '/training_defs/math09/*.xml.gz')
#xml_lst += glob('/media/hd1/training_defs/math14/*.xml.gz')
stream = stream_arxiv_paragraphs(xml_lst, samples=cfg['batch_size'])

all_data = []
for s in stream:
    all_data += list(zip(s[0], s[1]))
shuffle(all_data)


S = 5000 ## size of the test and validation sets
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

# +
#tknr = Tokenizer()
#tknr.fit_on_texts(list(validation[0]) + training[0])
#tot_words = len(tknr.word_index) + 1
#print(f"There is a total of {tot_words}")
#train_seq = tknr.texts_to_sequences(training[0])
#validation_seq = tknr.texts_to_sequences(validation[0])
# -

print('Starting the embedding matrix')
embed_matrix = np.zeros((cfg['tot_words'], 200))
coverage_cnt = 0
with open_w2v(base_dir + '/embeddings/model14-14_12-08/vectors.bin') as embed_dict:
    for word, ind in tkn2idx.items():
        vect = embed_dict.get(word)
        if vect is not None:
            #vect = vect/np.linalg.norm(vect)
            embed_matrix[ind] = vect
            coverage_cnt += 1

print("The coverage percetage is: {}".format(coverage_cnt/len(idx2tkn)))


lstm_model = Sequential([
    Embedding(cfg['tot_words'], 200, 
              input_length=max_seq_len,
              weights=[embed_matrix],
              trainable=False),
    Bidirectional(LSTM(128, return_sequences=True)),
    GlobalAveragePooling1D(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.summary()
history = lstm_model.fit(train_seq, np.array(training[1]),
                epochs=2, validation_data=(validation_seq, np.array(validation[1])),
                batch_size=512,
                verbose=1)


f1_max = 0.0; opt_prob = None
pred_validation = lstm_model.predict(validation_seq)

for thresh in np.arange(0.1, 0.901, 0.01):
    thresh = np.round(thresh, 2)
    f1 = metrics.f1_score(validation[1], (pred_validation > thresh).astype(int))
    #print('F1 score at threshold {} is {}'.format(thresh, f1))
    
    if f1 > f1_max:
        f1_max = f1
        opt_prob = thresh

print('Optimal probabilty threshold is {} for maximum F1 score {}'.format(opt_prob, f1_max))

pred_test = lstm_model.predict(test_seq)

print(metrics.classification_report((pred_test > opt_prob).astype(int), test[1]))

hoy = dt.now()
timestamp = hoy.strftime("%H-%M_%b-%d")
lstm_model.save_weights(base_dir + '/trained_models/lstm_classifier/one_layer_'+timestamp)

