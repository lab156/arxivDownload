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
import xml.etree.ElementTree as ET
from lxml import etree
import random
import sys
import glob
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
import collections as col
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import os.path
import gzip

# %load_ext autoreload
# %autoreload 2
import parsing_xml as px

stop_words = set(stopwords.words('english'))
# -

stacks_path = '/media/hd1/stacks-project/stacks-processed/'
ns = {'latexml': 'http://dlmf.nist.gov/LaTeXML' }

all_defs = []
root = etree.Element('root')
for xml_f in glob.glob(stacks_path + '*.xml'):
    try:
        DD = px.StacksProjectXML(xml_f)
        root.append(DD.create_xml_branch())
    except ValueError as e:
        print("Parse Error: ", e)

# + magic_args="echo this writes to disk" language="script"
# with gzip.open('/media/hd1/stacks-project/datasets/stacks-definitions.xml.gz', 'w') as fobj:
#     fobj.write(etree.tostring(root, pretty_print=True))
# -

print(etree.tostring(DD.create_xml_branch(), pretty_print=True).decode('utf8'))


