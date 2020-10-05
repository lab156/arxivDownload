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


# +
# random.random?

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

print(etree.tostring(Def.root, pretty_print=True).decode())

len(Def.root)

L = [(i,k) for i,k in enumerate(map(lambda s: s**2, range(10)))]
list(zip(*(L)))[1]


