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

from classifier_trainer.trainer import stream_arxiv_paragraphs
# -

rng = np.random.RandomState(0)
hash_param_grid = {'decode_error':'ignore',
              'n_features': [2 ** 21, 2 ** 22, 2 ** 23, 2**24],
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
cfg_param_grid = {'batch_size': [12500, 25000, 50000]}
HashSampler = ParameterSampler(hash_param_grid, n_iter=1)
clfSampler = ParameterSampler(clf_param_grid, n_iter=50000000)
cfgSampler = ParameterSampler(cfg_param_grid,   n_iter=1, random_state=rng)

for _ in range(15):
    print(next(cfgSampler.__iter__()))
#for p in cfgSampler:
    #print(p)

# +
# 2^18 = 262,144
# 2^21 = 2,097,152
###### CONFIG #####
xml_lst = glob("/media/hd1/training_defs/math*/*.xml.gz")
# THIS CONFIGURATION WORKS BEAUTIFULLY BUT STILL TRYING TO MAKE IT BETTER
#cfg = {'batch_size': 25000,
#      'hash_vect': {'decode_error':'ignore',
#                    'n_features': 2 ** 23,
#                    'alternate_sign': False,
#                    'ngram_range': (1,3)}, }
#hash_param = next(HashSampler.__iter__())
hash_param = {'stop_words': None,
              'norm': 'l2',
               'ngram_range': (1, 3),
              'n_features': 2097152,
               'lowercase': False,
               'decode_error': 'r',
               'binary': True,
               'analyzer': 'word',
               'alternate_sign': False}
vectorizer = HashingVectorizer(**hash_param)
cfg = {'batch_size': 5000}
logs_file = '../sgd_log.txt'

#clf_param = next(clfSampler.__iter__())
clf_param = {'alpha': 1.0329770832803815e-05,
             'eta0': 0.9052428812490275,
              'learning_rate': 'optimal',
              'loss': 'log',
              'penalty': 'elasticnet',
              'shuffle': True}
# Here are some classifiers that support the `partial_fit` method
partial_fit_classifiers = {
    'SGD': SGDClassifier(**clf_param),
    #'Perceptron': Perceptron(),
    #'NB Multinomial': MultinomialNB(alpha=0.01),
    #'Passive-Aggressive': PassiveAggressiveClassifier(),
}

# test data statistics
test_stats = {'n_test': 0, 'n_test_pos': 0}

# First we hold out a number of examples to estimate accuracy
stream = stream_arxiv_paragraphs(xml_lst, samples=cfg['batch_size'])
tick = time.time()
X_test_text, y_test = next(stream)
parsing_time = time.time() - tick
tick = time.time()
X_test = vectorizer.transform(X_test_text)
vectorizing_time = time.time() - tick
test_stats['n_test'] += len(y_test)
test_stats['n_test_pos'] += sum(y_test)
print("Test set is %d documents (%d positive)" % (len(y_test), sum(y_test)))


def progress(cls_name, stats):
    """Report progress information, return a string."""
    duration = time.time() - stats['t0']
    s = "%s classifier : " % cls_name
    s += "%(n_train)6d train docs (%(n_train_pos)6d positive) " % stats
    s += "%(n_test)6d test docs (%(n_test_pos)6d positive) " % test_stats
    s += "accuracy: %(accuracy).3f " % stats
    s += "in %.2fs (%5d docs/s)" % (duration, stats['n_train'] / duration)
    return s


cls_stats = {}

for cls_name in partial_fit_classifiers:
    stats = {'n_train': 0, 'n_train_pos': 0,
             'accuracy': 0.0, 'accuracy_history': [(0, 0)], 't0': time.time(),
             'runtime_history': [(0, 0)], 'total_fit_time': 0.0}
    cls_stats[cls_name] = stats

#get_minibatch(data_stream, n_test_documents)
# Discard test set
# -

total_vect_time = 0.0
# Main loop : iterate on mini-batches of examples
for i, (X_train_text, y_train) in enumerate(stream):

    tick = time.time()
    X_train = vectorizer.transform(X_train_text)
    total_vect_time += time.time() - tick

    for cls_name, cls in partial_fit_classifiers.items():
        tick = time.time()
        # update estimator with examples in the current mini-batch
        cls.partial_fit(X_train, y_train, classes=np.array([0,1]))

        # accumulate test accuracy stats
        cls_stats[cls_name]['total_fit_time'] += time.time() - tick
        cls_stats[cls_name]['n_train'] += X_train.shape[0]
        cls_stats[cls_name]['n_train_pos'] += sum(y_train)
        tick = time.time()
        cls_stats[cls_name]['accuracy'] = cls.score(X_test, y_test)
        cls_stats[cls_name]['prediction_time'] = time.time() - tick
        acc_history = (cls_stats[cls_name]['accuracy'],
                       cls_stats[cls_name]['n_train'])
        cls_stats[cls_name]['accuracy_history'].append(acc_history)
        run_history = (cls_stats[cls_name]['accuracy'],
                       total_vect_time + cls_stats[cls_name]['total_fit_time'])
        cls_stats[cls_name]['runtime_history'].append(run_history)

        if i % 3 == 0:
            print(progress(cls_name, cls_stats[cls_name]))
            if logs_file:
                with open(logs_file, 'a') as logs_fobj:
                    print(progress(cls_name, cls_stats[cls_name]),'\n', file = logs_fobj)
    if i % 3 == 0:
        print('\n')

# %%time
predictions = cls.predict(X_test)
print(metrics.classification_report(predictions,y_test))
auc_roc = metrics.roc_auc_score(predictions,y_test)
print(f'AUC-ROC: {auc_roc:3.2f}')


# +
def plot_accuracy(x, y, x_legend):
    """Plot accuracy as a function of x."""
    x = np.array(x)
    y = np.array(y)
    plt.title('Classification accuracy as a function of %s' % x_legend)
    plt.xlabel('%s' % x_legend)
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.plot(x, y)


rcParams['legend.fontsize'] = 10
rcParams["figure.figsize"] = [9.4, 6.8]
cls_names = list(sorted(cls_stats.keys()))

# Plot accuracy evolution
plt.figure()
for _, stats in sorted(cls_stats.items()):
    # Plot accuracy evolution with #examples
    accuracy, n_examples = zip(*stats['accuracy_history'])
    plot_accuracy(n_examples, accuracy, "training examples (#)")
    ax = plt.gca()
    ax.set_ylim((0.7, 1))
plt.legend(cls_names, loc='best')

plt.figure()
fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:,1], pred_prob)
plt.plot(fpr, tpr, lw=3)
#plt.figure()
#for _, stats in sorted(cls_stats.items()):
#    # Plot accuracy evolution with runtime
#    accuracy, runtime = zip(*stats['runtime_history'])
#    plot_accuracy(runtime, accuracy, 'runtime (s)')
#    ax = plt.gca()
#    ax.set_ylim((0.8, 1))
#plt.legend(cls_names, loc='best')

# Plot fitting times
#plt.figure()
#fig = plt.gcf()
#cls_runtime = [stats['total_fit_time']
#               for cls_name, stats in sorted(cls_stats.items())]

#cls_runtime.append(total_vect_time)
#cls_names.append('Vectorization')
#bar_colors = ['b', 'g', 'r', 'c', 'm', 'y']

#ax = plt.subplot(111)
#rectangles = plt.bar(range(len(cls_names)), cls_runtime, width=0.5,
#                     color=bar_colors)

#ax.set_xticks(np.linspace(0, len(cls_names) - 1, len(cls_names)))
#ax.set_xticklabels(cls_names, fontsize=10)
#ymax = max(cls_runtime) * 1.2
#ax.set_ylim((0, ymax))
#ax.set_ylabel('runtime (s)')
#ax.set_title('Training Times')


def autolabel(rectangles):
    """attach some text vi autolabel on rectangles."""
    for rect in rectangles:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.,
                1.05 * height, '%.4f' % height,
                ha='center', va='bottom')
        plt.setp(plt.xticks()[1], rotation=30)


#autolabel(rectangles)
#plt.tight_layout()
#plt.show()

# Plot prediction times
#plt.figure()
#cls_runtime = []
#cls_names = list(sorted(cls_stats.keys()))
#for cls_name, stats in sorted(cls_stats.items()):
#    cls_runtime.append(stats['prediction_time'])
#cls_runtime.append(parsing_time)
#cls_names.append('Read/Parse\n+Feat.Extr.')
#cls_runtime.append(vectorizing_time)
#cls_names.append('Hashing\n+Vect.')
#
#ax = plt.subplot(111)
#rectangles = plt.bar(range(len(cls_names)), cls_runtime, width=0.5,
#                     color=bar_colors)
#
#ax.set_xticks(np.linspace(0, len(cls_names) - 1, len(cls_names)))
#ax.set_xticklabels(cls_names, fontsize=8)
#plt.setp(plt.xticks()[1], rotation=30)
#ymax = max(cls_runtime) * 1.2
#ax.set_ylim((0, ymax))
#ax.set_ylabel('runtime (s)')
#ax.set_title(f'Prediction Times ({cfg["stream_samples"]:,d} instances)')
#autolabel(rectangles)
#plt.tight_layout()
#plt.show()
# -

with open('/media/hd1/PickleJar/sgd_clf_21-44_28-07.pickle', 'rb') as pickle_fobj:
    cls = pickle.load(pickle_fobj)
with open('/media/hd1/PickleJar/hash_vect_21-44_28-07.pickle', 'rb') as pickle_fobj:
    vectorizer = pickle.load(pickle_fobj)

tar_tree = etree.parse('/media/hd1/training_defs/math99/9902_001.xml.gz')
def_lst = tar_tree.findall('.//definition')
nondef_lst = tar_tree.findall('.//nondef')
ex_def = [D.text for D in def_lst[:15]]
ex_nondef = [D.text for D in nondef_lst[:15]]
preds_nondef = cls.predict(vectorizer.transform(ex_nondef))
preds_def = cls.predict(vectorizer.transform(ex_def))
print(f"Should be all zero: {preds_nondef}")
print('\n'.join(repr(k)+' --- '+ex_nondef[k] for k in np.nonzero(preds_nondef)[0]))
print('\n')
print(f"Should be all one: {preds_def}")
print('\n'.join(repr(k)+' --- '+ex_def[k] for k in np.nonzero(preds_def-1)[0]))

predictions = cls.predict(vectorizer.transform(X_test_text))
print(metrics.classification_report(predictions,y_test))
