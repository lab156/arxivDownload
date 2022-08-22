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
import numpy as np
import pandas as pd
from datetime import datetime
#import seaborn as sns
import matplotlib.pyplot as plt

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

from sklearn.preprocessing import LabelEncoder

#train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('../input/test.csv')

import nltk
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline  
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import sys
import re
import time
import glob
import gzip
from lxml import etree
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
import nltk.data
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
import pickle
from collections import Iterable
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.chunk import conlltags2tree, tree2conlltags
from sklearn.metrics import accuracy_score, log_loss
import random
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    naive_bayes.MultinomialNB(),
    KNeighborsClassifier(2),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    #GaussianNB(),
    #LinearDiscriminantAnalysis(),
    #QuadraticDiscriminantAnalysis(),
]

log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

#Local imports
# %load_ext autoreload
# %autoreload 2
from unwiki import unwiki
import ner
import parsing_xml as px
# -

stats = {}
time1 = time.time()
#xml_lst = glob.glob("/mnt/training_defs/math1*/*.xml.gz")
xml_lst = glob.glob("/media/hd1/training_defs/math17/*.xml.gz")
#allData = pd.DataFrame()
all_data_texts = []
all_data_labels = []
def_cnt = 0
nondef_cnt = 0
for X in xml_lst:
    tar_tree = etree.parse(X)
    def_lst = tar_tree.findall('.//definition')
    nondef_lst = tar_tree.findall('.//nondef')
    all_data_texts += [D.text for D in def_lst]
    all_data_labels += len(def_lst)*[1.0]
    all_data_texts += [D.text for D in nondef_lst]
    all_data_labels += len(nondef_lst)*[0.0]
    def_cnt += len(def_lst)
    nondef_cnt += len(nondef_lst)
stats['parsing_time'] = time.time() - time1
print("Definition count: {0:,d}.   NonDefinitions count: {1:,d}. Total: {2:,d}".format(def_cnt, nondef_cnt, (def_cnt+nondef_cnt)))
print('took {0:1.1f} secs'.format(stats['parsing_time']))

parags_lengths = [min(len(p.split()),400) for p in all_data_texts]
ex = plt.hist(parags_lengths, bins=100)
plt.xlabel('Word Count in Training Sample')
plt.ylabel('Number of Samples')
plt.title("Number of Words in Training Samples")
plt.grid()
plt.savefig('/home/luis/ims/train_length.png')

xml_lst = glob.glob("/mnt/training_defs/math9*/*.xml.gz")
def stream_arxiv_paragraphs(samples=1000):
    data_texts = []
    data_labels = []
    cnt = {'defs': 0, 'nondefs': 0}
    for X in xml_lst:
        tar_tree = etree.parse(X)
        def_lst = tar_tree.findall('.//definition')
        nondef_lst = tar_tree.findall('.//nondef')
        data_texts += [D.text for D in (def_lst + nondef_lst)] 
        data_labels += (len(def_lst)*[1.0] + len(nondef_lst)*[0.0])
        cnt['defs'] += len(def_lst)
        cnt['nondefs'] += len(nondef_lst)
        
        if cnt['defs'] + cnt['nondefs'] > samples:
            out_lst = list(zip(data_texts, data_labels))
            random.shuffle(out_lst)
            yield list(zip(*out_lst))
            data_texts = []
            data_labels = []
            cnt = {'defs': 0, 'nondefs': 0}
    return


# +
# define Clean function to cleanse and standarize words
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalize


#prepare the dataset Using 2018 old dataset
#allData = pd.DataFrame()
#with open('../sample18/defs.txt','r') as f1:
#    all_data_texts = f1.readlines()
#all_data_labels = len(all_data_texts)*[1.0]
#with open('../sample18/nondefs.txt', 'r') as f2:
#    all_data_texts_rand = f2.readlines()
#all_data_texts += all_data_texts_rand
#all_data_labels += len(all_data_texts_rand)*[0.0]

# 1.0 will represent definitions is true 0.0 means it is false (not a definition)
#allData['labels'] = all_data_labels
#allData['texts'] = all_data_texts

# Split and randomize the datasets
train_x, test_x, train_y, test_y = model_selection.train_test_split(all_data_texts, all_data_labels)

time1 = time.time()
# Vectorize all the paragraphs and definitions in the dataset
count_vect = CountVectorizer(max_features=50000,analyzer='word', tokenizer=nltk.word_tokenize, ngram_range=(1,3))
count_vect.fit(all_data_texts)
stats['vectorizer_time'] = time.time() - time1
xtrain = count_vect.transform(train_x)
xtest = count_vect.transform(test_x)

# Train Multinomial Naive Bayes model and print test metrics
#clf = naive_bayes.MultinomialNB().fit(xtrain, train_y)
#predictions = clf.predict(xtest)
#print(metrics.classification_report(predictions,test_y))
print(f"vectorizer time: {stats['vectorizer_time']:1.2f}")
print(f"Train size: {len(train_x):,d}. Test size {len(test_x)}. Total {len(test_x)+len(train_x):,d}")
print(f"Vocab len: {len(count_vect.vocabulary_):,d}")
# -

with open('/media/hd1/PickleJar/count_vectorizer49.pickle', 'rb') as pickle_obj:
    count_vect = pickle.load(pickle_obj)
with open('/media/hd1/PickleJar/classifier49.pickle', 'rb') as pickle_obj:
    clf = pickle.load(pickle_obj)

#param_lst = [1550, 1650, 1750]
#SVC_lst = [SVC(kernel="rbf", C=param, probability=True) for param in param_lst]
#classifiers=  [ naive_bayes.MultinomialNB(),]
cfg={'kernel':"rbf", 'C':1600, 'probability':True}
classifiers=  [SVC(**cfg)]
classifiers[0].__class__.__name__

# +
#for C_param, clf in zip(param_lst,SVC_lst):
for C_param, clf in enumerate(classifiers):
    name = clf.__class__.__name__
    clf.fit(xtrain, train_y)
    
    print("="*30)
    print(name, ',  C=', C_param)
    
    print('****Results****')
    predictions = clf.predict(xtest)
    acc = accuracy_score(test_y, predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    #prodictions = prob_prediction
    prodictions = clf.predict_proba(xtest)
    ll = log_loss(test_y, prodictions)
    print("Log Loss: {}".format(ll))
    
    #log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    #log = log.append(log_entry)
    
print("="*30)
# -

# %%time
stream = stream_arxiv_paragraphs(samples=25)
test_x, test_y = next(stream)
predictions = clf.predict(count_vect.transform(test_x))
print(metrics.classification_report(predictions,test_y))

# %%time
Def = ['a banach space is defined as a complete vector space.',
       'This is not a definition honestly. even if it includes technical words like scheme and cohomology',
      'There is no real reason as to why this classifier is so good.',
      'a triangle is equilateral if and only if all its sides are the same length.',
      ' The paper is organized as follows. ',
      'Proof. By definition (6.4) _display_math_ where _inline_math_ denotes the parity of _inline_math_ and _display_math_',
      'Counting subobjects over finite fields, as in Ringel _citation_.']
vdef = count_vect.transform(Def)
clf.predict(vdef)

tar_tree = etree.parse('/media/hd1/training_defs/math19/1902_001.xml.gz')
def_lst = tar_tree.findall('.//definition')
nondef_lst = tar_tree.findall('.//nondef')
ex_def = [D.text for D in def_lst[:15]]
ex_nondef = [D.text for D in nondef_lst[:15]]
preds_nondef = clf.predict(count_vect.transform(ex_nondef))
preds_def = clf.predict(count_vect.transform(ex_def))
print(f"Should be all zeros: {preds_nondef}")
print('Showing the false positives')
print('\n'.join(repr(k)+' --- '+ex_nondef[k] for k in np.nonzero(preds_nondef)[0]))
print('-------------------------------------------')
print(f"Should be all ones: {preds_def}")
print('Showing the false negatives')
print('\n'.join(repr(k)+' --- '+ex_def[k] for k in np.nonzero(preds_def-1)[0]))

#pred_prob = clf.predict_proba(count_vect.transform(test_x))
y_test_array = pd.get_dummies(test_y, drop_first=False).values
plt.figure()
fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_prob[:,1])
plt.plot(fpr, tpr, lw=3)

pred_prob[:,1].shape


