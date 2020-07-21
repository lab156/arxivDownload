from glob import glob
import itertools
import os.path
import re
import tarfile
import time
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from html.parser import HTMLParser
from urllib.request import urlretrieve
import sklearn.metrics as metrics
from sklearn.datasets import get_data_home
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from scipy.stats import uniform

from lxml import etree
from random import shuffle

from trainer import stream_arxiv_paragraphs

cfg = {'train_data': "/home/lab232/training_defs/math18/*.xml.gz"}  
hash_param_grid = { 'n_features': [2 ** 21, 2 ** 22, 2 ** 23, 2**24],
              'alternate_sign': [False, True],
              'ngram_range': [(1,2), (1,3)],
             'binary': [False, True],
             'norm': ['l1', 'l2'],
             'analyzer': ['word', 'char', 'char_wb'],
             'stop_words': ['english', None],
             'lowercase': [False, True],
             }
clf_param_grid = {'loss': ['hinge', 'log', 'squared_hinge', 'modified_huber'],
              'penalty': ['l2', 'l1', 'elasticnet'],
              'alpha': uniform(0.0001/10, 0.0001*1.4),
              'shuffle': [False, True],
              'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                 'eta0': uniform(0.01, 1),}
cfg_param_grid = {'batch_size': [12500, 25000, 50000], }
HashSampler = ParameterSampler(hash_param_grid, n_iter=1)
clfSampler = ParameterSampler(clf_param_grid,   n_iter=1)
cfgSampler = ParameterSampler(cfg_param_grid,   n_iter=1)

tboy_acc = 0
cnt = 0
while True:
    xml_lst = glob(cfg['train_data'])
    hash_param = next(HashSampler.__iter__())
    vectorizer = HashingVectorizer(**hash_param)

    clf_param = next(clfSampler.__iter__())
    # Here are some classifiers that support the `partial_fit` method
    partial_fit_classifiers = {
        'SGD': SGDClassifier(**clf_param),
    }

    # test data statistics
    test_stats = {'n_test': 0, 'n_test_pos': 0}

    # First we hold out a number of examples to estimate accuracy
    cfg_param = next(cfgSampler.__iter__())
    stream = stream_arxiv_paragraphs(xml_lst, samples=cfg_param['batch_size'])
    X_test_text, y_test = next(stream)
    X_test = vectorizer.transform(X_test_text)

    total_vect_time = 0.0
    # Main loop : iterate on mini-batches of examples
    for i, (X_train_text, y_train) in enumerate(stream):

        X_train = vectorizer.transform(X_train_text)

        for cls_name, cls in partial_fit_classifiers.items():
            # update estimator with examples in the current mini-batch
            cls.partial_fit(X_train, y_train, classes=np.array([0,1]))

    temp_acc = cls.score(X_test, y_test)
    if temp_acc > tboy_acc:
        tboy_acc = temp_acc
        print(f'''Found a better set of params with acc: {temp_acc}
        Batch size: {cfg_param["batch_size"]} at iteration: {cnt}
        Hash Parameters: {hash_param} 
        Classifier Params: {clf_param}
        ---------------------

        ''')
    cnt += 1

