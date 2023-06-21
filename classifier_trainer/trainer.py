import numpy as np
import os
#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
from datetime import datetime

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

from sklearn.preprocessing import LabelEncoder

#train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('../input/test.csv')

import nltk
import pandas as pd
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
import socket
import gzip
import logging
from lxml import etree
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
import nltk.data
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
import pickle
from collections.abc import Iterable
from nltk.tag import ClassifierBasedTagger
from nltk.chunk import ChunkParserI
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.chunk import conlltags2tree, tree2conlltags
from sklearn.metrics import accuracy_score, log_loss
import random
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC #, LinearSVC, NuSVC
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#classifiers = [
#    naive_bayes.MultinomialNB(),
#    KNeighborsClassifier(2),
#    SVC(kernel="rbf", C=0.025, probability=True),
#    NuSVC(probability=True),
#    DecisionTreeClassifier(),
#    RandomForestClassifier(),
#    AdaBoostClassifier(),
#    GradientBoostingClassifier(),
#    #GaussianNB(),
#    #LinearDiscriminantAnalysis(),
#    #QuadraticDiscriminantAnalysis(),
#]

#Local imports
#from unwiki import unwiki
#import ner
#import parsing_xml as px

def stream_arxiv_paragraphs(xml_lst, samples=1000):
    '''
    xml_lst is a list of file
    xml_lst = glob("/mnt/training_defs/math1*/*.xml.gz")
    iterate batches of paragraphs of size `samples` 
    returns in format x_train (list of texts), y_label (list of labels 0 & 1)
    '''
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

        if cnt['defs'] + cnt['nondefs'] > samples or X == xml_lst[-1]:
            out_lst = list(zip(data_texts, data_labels))
            random.shuffle(out_lst)
            logging.debug('Definition count: defs: {defs} nondefs: {nondefs}'.format(**cnt))
            yield list(zip(*out_lst))
            data_texts = []
            data_labels = []
            cnt = {'defs': 0, 'nondefs': 0}
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Classifier and Vectorize')
    parser.add_argument('xmlpath', type=str, nargs='+',
            help='Paths to the training data xml files ex. /mnt/training_defs/math1*/*.xml.gz')
    parser.add_argument('savedir', type=str,
            default='/mnt/PickleJar/trainer_datalog/',
            help='Complete path to the directory to save. /mnt/PickleJar/trainer_datalog/')
    parser.add_argument('--samples', type=int, 
            default=200,
            help='Minimum size of batches of paragraphs to use for training')
    parser.add_argument('--log', type=str, default="info",
            help='Log level warning, info, debug, etc.')
    args = parser.parse_args(sys.argv[1:])

    cfg = {'save_dir': args.savedir,
            'n_samples': args.samples, }

    # Create save_dir if not exists
    os.makedirs(cfg['save_dir'], exist_ok=True)

    logging.basicConfig(filename=os.path.join(cfg['save_dir'], 'classifier_trainer.log'),
            level=getattr(logging, args.log.upper()))

    # Get the data from the sample text

    #xml_lst = glob.glob("/mnt/training_defs/math1*/*.xml.gz")
    xml_lst = args.xmlpath
    logging.debug(f'xml_lst is: {xml_lst[:3]}')
    logging.debug('The number of samples is: {n_samples}'.format(**cfg))
    stream = stream_arxiv_paragraphs(xml_lst, samples=cfg['n_samples'])
    data_texts, data_labels = next(stream)

    cfg['test_size'] = 2500 if len(data_labels) > 150000 else 0.25

    train_x, test_x, train_y, test_y = model_selection.train_test_split(
            data_texts, data_labels, test_size=cfg['test_size'])
    cfg['train_size'] = len(train_x)
    time1 = time.time()
    # Vectorize all the paragraphs and definitions in the dataset
    cfg['vect'] = {'max_features': 50000,
            'analyzer': 'word',
            'tokenizer': nltk.word_tokenize,
            'ngram_range': (1,3)}
    count_vect = CountVectorizer(**cfg['vect'])
    count_vect.fit(data_texts)
    cfg['vectorizer_time'] = time.time() - time1

    cfg['n_vocab'] = len(count_vect.vocabulary_)
    cfg['n_parag'] = len(data_labels)


    # vectorize the train and test texts
    xtrain = count_vect.transform(train_x)
    xtest = count_vect.transform(test_x)

    # Define the classifier with its parameters
    cfg['clf'] = {'kernel': 'rbf',
            'C': 1600,
            'probability': True,}
    clf = SVC(**cfg['clf'])
    cfg['clf_name'] = clf.__class__.__name__

    # Train the classifier
    time1 = time.time()
    clf.fit(xtrain, train_y)
    cfg['train_time'] = time.time() - time1

    predictions = clf.predict(xtest)
    cfg['clf_acc'] = accuracy_score(test_y, predictions)

    cfg['host'] = socket.gethostname()
    cfg['timestamp'] = datetime.now().strftime("%H-%M_%d-%m")

    logging.info("""
    Timestamp: {timestamp} on {host}
    Time vectorizing: {vectorizer_time:3.1f}
    Time spent training: {train_time:3.1f}
    Number of Paragraphs: {n_parag:,d} Training: {train_size:,d} Testing: {test_size:3.3f}
    Classifier Accuracy: {clf_acc:3.3f}
    """.format(**cfg))

    # Store the results
    with open(os.path.join(cfg['save_dir'], 'count_vect_'+cfg['timestamp']+'.pickle'), 'wb') as vect_fobj:
        pickle.dump(count_vect, vect_fobj)
    with open(os.path.join(cfg['save_dir'], 'clf_'+cfg['timestamp']+'.pickle'), 'wb') as clf_fobj:
        pickle.dump(clf, clf_fobj)

