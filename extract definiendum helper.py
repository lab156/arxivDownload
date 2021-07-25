# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import parsing_xml as px
import extract as X
import peep_tar as peep
from lxml import etree
import random
import numpy as np
import gzip
import os
from IPython.core.display import display, HTML
import pickle
from ner.chunker import NamedEntityChunker, features

# +
glossary_NN = '/home/luis/rm_me_glossary/test_conv/math01/'
glossary_sgd = '/media/hd1/glossary/v3/math01/'

N = 0 # total number of paragraphs to be classified
n_NN1 = 0 # number of definitions found by NN
n_NN0 = 0 # number of nondefintion found by NN
n_sgd1 = 0 # number of definitions found by sgd
n_sgd0 = 0 # number of nondefitions found by sgd
TP = 0 # True positives of agreement between classifiers
TN = 0 # True negatives 

idx_set = lambda tree: set(int(a.attrib['index']) for a in tree.findall('definition'))

for tfile in os.listdir(glossary_sgd):
    with gzip.open(glossary_sgd + tfile, 'r') as gzobj_sgd:
        with gzip.open(glossary_NN + tfile, 'r') as gzobj_NN:
            tree_sgd = etree.parse(gzobj_sgd)
            tree_NN = etree.parse(gzobj_NN)
            print(tfile, '  ', len(tree_sgd.findall('//article')), '  ', len(tree_NN.findall('//article')))
            #sgd_set = set(a.attrib['name'] for a in tree_sgd.findall('//article'))
            #nn_set = set(a.attrib['name'] for a in tree_NN.findall('//article'))
            para_num = sum(int(a.attrib['num']) for a in tree_sgd.findall('//article'))
            N += para_num
            n_NN1 += len(tree_NN.findall('//definition'))
            n_NN0 += para_num - n_NN1
            n_sgd1 += len(tree_sgd.findall('//definition'))
            n_sgd0 += para_num - n_sgd1
            
            NN_art_lst = tree_NN.findall('//article')
            sgd_art_lst = tree_sgd.findall('//article')
            
            for index in range(len(tree_NN.findall('//article'))):
                NN_set = idx_set(NN_art_lst[index])
                NN_0_set = set(range(int(NN_art_lst[index].attrib['num']))).difference(NN_set)
                sgd_set = idx_set(sgd_art_lst[index])
                sgd_0_set = set(range(int(sgd_art_lst[index].attrib['num']))).difference(sgd_set)
                TP += len(NN_set.intersection(sgd_set))
                TN += len(NN_0_set.intersection(sgd_0_set))
                
            
            
print(f'N = {N}')
po = (TP + TN)/N
pe = 1/N**2*(n_NN1*n_sgd1 + n_NN0*n_NN0)
kappa = (po - pe)/(1 - pe)
print(f"Cohen's kappa is: {kappa}")
# -

range_set = set(range(10))
range_set.difference(set(range(5)))


# ### Cohen's kappa interrater agreement statistic
# From https://en.wikipedia.org/wiki/Cohen%27s_kappa 
# For $N$ observations and the two categories: NN and sgd
# $$\kappa = \frac{p_o - p_e}{1- p_e}$$
# Where $p_o$ and $p_e$ are defined as:
# $$ p_o = \frac{TP + TN}{N} \quad p_e = \frac{1}{N^2}(n_1^{NN} n_1^{sgd} + n_0^{NN} n_0^{sgd})$$

def nice_print(index):
    idx_set = lambda tree: set(int(a.attrib['index']) for a in tree.findall('definition'))
    NN_set = idx_set(tree_NN.findall('//article')[index])
    sgd_set = idx_set(tree_sgd.findall('//article')[index])
    all_set = sgd_set.union(NN_set)
    color = {'both': "100,250,100", 'nn': "250,100,250", 'sgd': "300,100,100"}
    cfun = lambda rgb, idx: '<span style="background-color:rgba({1})";> &nbsp;  {0:>4} </span>'.format(idx,rgb)
    hl_text = []
    for i in sorted(all_set):
        if i in sgd_set and i in NN_set:
            hl_text.append(cfun(color['both'],i))
        elif i in sgd_set:
            hl_text.append(cfun(color['sgd'],i))
        else:
            hl_text.append(cfun(color['nn'],i))

    hl_text = ' '.join(hl_text)
    display(HTML(hl_text))
for i in range(20,80):
    nice_print(i)


# +
class FarseClf():
    def __init__(self):
        pass
    def predict(self, lst, **kwargs):
        return np.array([True for _ in lst])
    
class RandClf():
    def __init__(self):
        pass
    def predict(self, lst, **kwargs):
        np.random.seed(seed=42)
        return np.random.rand(len(lst))
    
class FarseVectorizer():
    def __init__(self):
        pass
    def transform(self, lst):
        return [0.0 for _ in lst]

class FarseBio():
    def __init__(self):
        pass
    def parse(self, lst):
        return [0.0 for _ in lst]


# -

clf = FarseClf()
vzer = FarseVectorizer()
bio = FarseBio()
for fname, tfobj in peep.tar_iter('tests/five_actual_articles.tar.gz', '.xml'):
    parsing_fobj = px.DefinitionsXML(tfobj)
    Def = X.Definiendum(parsing_fobj, clf, None, vzer, None, min_words=15 )

print(etree.tostring(xml.exml, pretty_print=True).decode())

[(idx,p) for idx,p in enumerate(map(xml.recutext, xml.para_list())) if len(p.split()) >= 15]

with open('/media/hd1/PickleJar/sgd_clf_21-44_28-07.pickle', 'rb') as class_f:
    clf = pickle.load(class_f)
with open('/media/hd1/PickleJar/chunker.pickle', 'rb') as class_f:
    bio = pickle.load(class_f)
with open('/media/hd1/PickleJar/hash_vect_21-44_28-07.pickle', 'rb') as class_f:
    vzer = pickle.load(class_f)
with open('/media/hd1/PickleJar/tokenizer.pickle', 'rb') as class_f:
    tokr = pickle.load(class_f)

# ## Cohen's Kappa history:
# * 0.9306055714816674: Single layer LSTM with .93 f1 trained on math01, math14, math15.
# * 0.9296264157862171: Single layer LSTM with .93 f1 trained on math14 and math15 __only__
# * 0.9294064163568799: Double layer LSTM with .93 f1 trained on math14 and math15 __only__
# * 0.9270823744884535: Covolutional 1D with .92 f1 trained math14 and math15 __only__

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info('macizo mac')

logger

dicti = {1: 'hola'}
[k for k,v in dicti.items()]


