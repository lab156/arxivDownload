from glob import glob
import os
import numpy as np
from lxml import etree
from collections import Counter
from random import shuffle
from datetime import datetime as dt
import logging
import gzip

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional,\
                      GRU, Dropout, GlobalAveragePooling1D, Conv1D
from tensorflow.keras.models import Sequential


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



# GET the Important Paths
base_dir = os.environ['PROJECT'] # This is permanent storage
local_dir = os.environ['LOCAL']  # This is temporary fast storage

cfg = {'batch_size': 5000,
      'glob_data_source': '/training_defs/math16/*.xml.gz',
      'TVT_split' : 0.8,    ## Train  Validation Test split
      'max_seq_len': 400,   # Length of padding and input of Embedding layer
      'promath_dir': 'promath', # name of dir with the processed arXiv tar files
      'save_path': 'glossary/test_lstm', #Path to save the positive results
      'min_words': 15, # min number of words for paragraphs to be considered
      }
xml_lst = glob(base_dir + cfg['glob_data_source'])
#xml_lst += glob('/media/hd1/training_defs/math14/*.xml.gz')

# CREATE LOG FILE AND OBJECT
hoy = dt.now()
timestamp = hoy.strftime("%b-%d_%H-%M")
save_path_dir = os.path.join(base_dir, 'trained_models/lstm_classifier/lstm_' + timestamp)
os.mkdir(save_path_dir)

logging.basicConfig(filename=os.path.join(save_path_dir, 'log.txt'),
        level=logging.INFO)
logger = logging.getLogger(__name__)


# READ THE TRAINING DATA
stream = stream_arxiv_paragraphs(xml_lst, samples=cfg['batch_size'])

all_data = []
for s in stream:
    all_data += list(zip(s[0], s[1]))
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
print("Most common tokens are:", tknr.most_common()[:10])

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


# +
def text2seq(text):
    if type(text) == str:
        text = normalize_text(text).split()
    return [tkn2idx.get(s, 0) for s in text]



#padding_fun = lambda seq: pad_sequences(seq, maxlen=cfg['max_seq_len'],
#                                        padding='post', 
#                                        truncating='post',
#                                        value=tkn2idx['�']) 

def padding_fun(seq, cfg):
    # Apply pad_sequence using the cfg dictionary
    return pad_sequences(seq, maxlen=cfg['max_seq_len'],
                            padding='post', 
                            truncating='post',
                            value=tkn2idx['�']) 

# CREATE THE DATA SEQUENCES
train_seq = [text2seq(t) for t in training[0]]
validation_seq = [text2seq(t) for t in validation[0]]
test_seq = [text2seq(t) for t in test[0]]

# PAD THE SEQUENCES AND KEEP THE VARIABLE NAME
train_seq = padding_fun(train_seq, cfg)
validation_seq = padding_fun(validation_seq, cfg)
test_seq = padding_fun(test_seq, cfg)

# +
#tknr = Tokenizer()
#tknr.fit_on_texts(list(validation[0]) + training[0])
#tot_words = len(tknr.word_index) + 1
#print(f"There is a total of {tot_words}")
#train_seq = tknr.texts_to_sequences(training[0])
#validation_seq = tknr.texts_to_sequences(validation[0])
# -

print('Starting the embedding matrix')
coverage_cnt = 0
with open_w2v(base_dir + '/embeddings/model14-14_12-08/vectors.bin') as embed_dict:
    cfg['embed_dim'] = embed_dict[next(iter(embed_dict))].shape[0]
    embed_matrix = np.zeros((cfg['tot_words'], cfg['embed_dim']))
    for word, ind in tkn2idx.items():
        vect = embed_dict.get(word)
        if vect is not None:
            #vect = vect/np.linalg.norm(vect)
            embed_matrix[ind] = vect
            coverage_cnt += 1

print("The coverage percentage is: {}".format(coverage_cnt/len(idx2tkn)))


def lstm_model_one_layer(cfg):
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

cfg['lstm_cells'] = 128 # Required LSTM layer parameter
lstm_model = lstm_model_one_layer(cfg)


cfg['model_name'] = lstm_model_one_layer.__name__

history = lstm_model.fit(train_seq, np.array(training[1]),
                epochs=2, validation_data=(validation_seq, np.array(validation[1])),
                batch_size=512,
                verbose=1)


# Find the best classification cutoff parameter
def find_best_cutoff(model, val_seq):
    '''
    model: is an instance with predict attribute
    val_seq: has same shape and format as the training sequence
    '''
    f1_max = 0.0; opt_prob = None
    pred_validation = lstm_model.predict(val_seq)

    for thresh in np.arange(0.1, 0.901, 0.01):
        thresh = np.round(thresh, 2)
        f1 = metrics.f1_score(validation[1], (pred_validation > thresh).astype(int))
        #print('F1 score at threshold {} is {}'.format(thresh, f1))
        
        if f1 > f1_max:
            f1_max = f1
            opt_prob = thresh
    return (opt_prob, f1_max)

opt_prob, f1_max = find_best_cutoff(lstm_model, validation_seq)

logger.info('\n Optimal probabilty threshold is {} for maximum F1 score {}\n'\
        .format(opt_prob, f1_max))

pred_test = lstm_model.predict(test_seq)

metrics_str = metrics.classification_report((pred_test > opt_prob).astype(int), test[1])
logger.info('\n' + metrics_str)



lstm_model.save_weights( os.path.join(save_path_dir, 'model_weights') )

# Log both a pretty printed and a copy-pasteable version of the the cfg
# dictionary
logger.info('\n'.join(["{}: {}".format(k,v) for k, v in cfg.items()]))
logger.info(repr(cfg))

##### Parameters of Features to explore
# length of padding (number of input tokens)
# length of lstm cells
# change the word embedding model, this should a list of available word embedding with a method to open them

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
    root = etree.Element('root')
    for fname, tar_fobj in peep.tar_iter(tfile, '.xml'):
        try:
            DD = px.DefinitionsXML(tar_fobj) 
            if DD.det_language() in ['en', None]:
                art_tree = Definiendum(DD, clf, None, vzer,\
                        None, fname=fname, thresh=opt_prob, min_words=cfg['min_words']).root
                if art_tree is not None: root.append(art_tree)
        except ValueError as ee:
            print(f"{repr(ee)}, 'file: ', {fname}, ' is empty'")
    return root
    
for k, dirname in enumerate(['math96',]):
#for k, dirname in enumerate(['math' + repr(k)[2:] for k in range(1996, 1994, 1)]):
    logger.info('Classifying the contents of {}'.format(dirname))
    try:
        full_path = os.path.join(base_dir, cfg['promath_dir'], dirname)
        tar_lst = [os.path.join(full_path, p) for p in os.listdir(full_path) if '.tar.gz' in p]
    except FileNotFoundError:
        print(' %s Not Found'%full_path)
        break
    out_path = os.path.join(local_dir, cfg['save_path'], dirname)
    os.makedirs(out_path, exist_ok=True)
   
    for tfile in tar_lst:
        #clf = lstm_model
        vzer = Vectorizer()
        def_root = untar_clf_append(tfile, out_path, lstm_model, vzer, thresh=opt_prob)
        #print(etree.tostring(def_root, pretty_print=True).decode())
        gz_filename = os.path.basename(tfile).split('.')[0] + '.xml.gz' 
        print(gz_filename)
        gz_out_path = os.path.join(out_path, gz_filename) 
        with gzip.open(gz_out_path, 'wb') as out_f:
            print("Writing to dfdum zipped file to: %s"%gz_out_path)
            out_f.write(etree.tostring(def_root, pretty_print=True))


