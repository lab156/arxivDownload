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

import parsing_xml as px
import extract as X
import peep_tar as peep
from lxml import etree
import random
import numpy as np
import gzip
import os
from IPython.core.display import display, HTML

glossary_NN = '/home/luis/rm_me_glossary/test/math01/'
glossary_sgd = '/media/hd1/glossary/v2.1/math01/'
for tfile in os.listdir(glossary_sgd):
    with gzip.open(glossary_sgd + tfile, 'r') as gzobj_sgd:
        with gzip.open(glossary_NN + tfile, 'r') as gzobj_NN:
            tree_sgd = etree.parse(gzobj_sgd)
            tree_NN = etree.parse(gzobj_NN)
            print(tfile, '  ', len(tree_sgd.findall('//article')), '  ', len(tree_NN.findall('//article')))

for d in tree.findall('//article')[0]:
    print(d.attrib['index'])

art_tree = tree.findall('//article')[0]
idx_lst = [a.attrib['index'] for a in art_tree.findall('definition')]
hl_text = []
for i in idx_lst:
    w = 0.5
    hl_text.append('<span style="background-color:rgba(135,205,250)";> &nbsp;  {0:>4} </span>'.format(i))
hl_text = ' '.join(hl_text)
display(HTML(hl_text))

print(hl_text)


# +
class FarseClf():
    def __init__(self):
        pass
    def predict(self, lst):
        return [True for _ in lst]
    
class RandClf():
    def __init__(self):
        pass
    def predict(self, lst):
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

clf = RandClf()
vzer = FarseVectorizer()
bio = FarseBio()
for fname, tfobj in peep.tar_iter('tests/five_actual_articles.tar.gz', '.xml'):
    parsing_fobj = px.DefinitionsXML(tfobj)
    Def = X.Definiendum(parsing_fobj, clf, None, vzer, None, min_words=40, thresh=0.96)

# + jupyter={"outputs_hidden": true}
print(etree.tostring(Def.root, pretty_print=True).decode())
# -

len(Def.root)

L = [(i,k) for i,k in enumerate(map(lambda s: s**2, range(10)))]
list(zip(*(L)))[1]


