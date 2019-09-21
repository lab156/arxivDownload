# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import parsing_xml as px
import xml.etree.ElementTree as ET
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

stop_words = set(stopwords.words('english'))
# -

stacks_path = '../stacks-clean/'
ns = {'latexml': 'http://dlmf.nist.gov/LaTeXML' }

tnzer = RegexpTokenizer(r'\w+')
resu = tnzer.tokenize(all_defs[0])
resu 

# +
stop_words.update(set(['1', '2', '_inline_math_', '_display_math_']))

word_flt = list(filter(lambda w: w not in stop_words, tnzer.tokenize('\n'.join(plst_all))))
word_map = map(lemmatizer.lemmatize, word_flt)
# -

count = col.Counter(word_map)

for ph in count.most_common()[:300]:
    text = ph[0]
    freq = ph[1]
    print("{:<25} {:>5}".format(''.join(text), freq))

# +
lemmatizer = WordNetLemmatizer() 
  
print("continuity :", lemmatizer.lemmatize("continua")) 
print("corpora :", lemmatizer.lemmatize("corpora")) 
# -

all_defs = []
for xml_f in glob.glob('../stacks-clean/*.xml'):
    try:
        DD = px.DefinitionsXML(xml_f)
        def_lst = DD.get_def_text()
        all_defs += def_lst
    except ValueError as e:
        print("Parse Error: ", e)


def para_tags(f, ns, min_words=0):
    '''
    Usage: f be a parsable xml tree
    try to get para_per_article paragraphs from this article
    min_words: the paragraph has to have more that this amount of words
    '''
    try:
        exml = ET.parse(f)
        para_lst = exml.findall('.//latexml:para',ns)
    except ET.ParseError:
        print('article %s could no be parsed'%f)
        para_lst = []
    except ValueError:
        print('article %s has few paragraphs'%f)
        para_lst = []

    return_lst = []
    for p in para_lst:
        if px.check_sanity(p, ns):
            para_text =  px.recutext_xml(p)
            if len(para_text.split()) >= min_words: #check min_words
                return_lst.append(para_text)
        else:
            print('article %s has messed up para'%f)
    return return_lst


plist = para_tags(stacks_path+'perfect.xml', ns, min_words=15)

docu_lst = glob.glob('../stacks-clean/*.xml')
plst_all = sum(map(lambda D: para_tags(D, ns, min_words=15), docu_lst),[])

plst_all[2]

count = col.Counter(ngrams(' '.join(all_defs).split(),8))

for ph in count.most_common():
    text = ph[0]
    freq = ph[1]
    print("{:<65} {:>5}".format(' '.join(text), freq))

v = CountVectorizer(ngram_range=(1,1))
v.fit_transform([plist[2]])

for w in v.get_feature_names():
    print(w, v.vocabulary_[w])

v.vocabulary_['we']


