# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
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
from sklearn import metrics
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from ner.chunker import NamedEntityChunker, features
from lxml import etree
import tarfile
import logging
import os
import glob

#Local imports
# %load_ext autoreload
# %autoreload 2
from unwiki import unwiki
import ner
import parsing_xml as px
import mp_extract

# +
with open('/mnt/promath/math05/0503_001/math.0503029/math.0503029.xml', 'r') as art_file:
    root = mp_extract.parse_clf_chunk(art_file, clf, bio, vzer, tokr)
    
with open('/home/pi/rm_me_please.gz', 'wb') as pl:
    st = etree.tostring(root, pretty_print=True)
    pl.write(st)
# -

glob.glob('/mnt/promath/math15/1501_00*.tar.gz')

clsf = '/mnt/PickleJar/classifier.pickle'
chun = '/mnt/PickleJar/chunker.pickle'
vect = '/mnt/PickleJar/vectorizer.pickle'
toke = '/mnt/PickleJar/tokenizer.pickle'
logging.basicConfig(level = logging.DEBUG)
with open(clsf, 'rb') as class_f:                                                                                        
    clf = pickle.load(class_f)                                                                                                      
with open(chun, 'rb') as class_f:
    bio = pickle.load(class_f)
with open(vect, 'rb') as class_f:                                                                                        
    vzer = pickle.load(class_f)
with open(toke, 'rb') as class_f:                                                                                         
    tokr = pickle.load(class_f)  
#with tarfile.open('/mnt/promath/math15/1501_001.tar.gz', 'r') as art_file:
    #root = mp_extract.parse_clf_chunk(art_file, clf, bio, vzer, tokr)
#    print(art_file.getnames())
#print(etree.tostring(root, pretty_print=True))
mp_extract.untar_clf_write('/mnt/promath/math15/1501_001.tar.gz', '/home/pi/hola_rm_me', clf, bio, vzer, tokr)

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

#prepare the dataset
allData = pd.DataFrame()
with open('../sample18/defs.txt','r') as f1:
    all_data_texts = f1.readlines()
all_data_labels = len(all_data_texts)*[1.0]
with open('../sample18/nondefs.txt', 'r') as f2:
    all_data_texts_rand = f2.readlines()
all_data_texts += all_data_texts_rand
all_data_labels += len(all_data_texts_rand)*[0.0]

# 1.0 will represent definitions is true 0.0 means it is false (not a definition)
allData['labels'] = all_data_labels
allData['texts'] = all_data_texts

# Split and randomize the datasets
train_x, test_x, train_y, test_y = model_selection.train_test_split(allData['texts'], allData['labels'])

# Vectorize all the paragraphs and definitions in the dataset
count_vect = CountVectorizer(analyzer='word', tokenizer=nltk.word_tokenize, ngram_range=(1,2))
count_vect.fit(allData['texts'])
xtrain = count_vect.transform(train_x)
xtest = count_vect.transform(test_x)

# Train Multinomial Naive Bayes model and print test metrics
clf = naive_bayes.MultinomialNB().fit(xtrain, train_y)
predictions = clf.predict(xtest)
print(metrics.classification_report(predictions,test_y))
# -

Def = ['a banach space is defined as a complete vector space.',
       'This is not a definition honestly. even if it includes technical words like scheme and cohomology',
      'There is no real reason as to why this classifier is so good.',
      'a triangle is equilateral if and only if all its sides are the same length.']
vdef = count_vect.transform(Def)
clf.predict(vdef)

# +
# The results for the search for definition (currently just Wikipedia)
with open('../wiki_definitions_improved.txt', 'r') as wiki_f:
    wiki = wiki_f.readlines()
    
# Get data and train the Sentence tokenizer
# Splits the individual sentences of a paragraph apart 
# Uses a standard algorithm (Kiss-Strunk) for unsupervised sentence boundary detection
text = ''
for i in range(550):
    text += unwiki.loads(eval(wiki[i].split('-#-%-')[2]))

trainer = PunktTrainer()
trainer.INCLUDE_ALL_COLLOCS = True
trainer.train(text)
tokenizer = PunktSentenceTokenizer(trainer.get_params())
#print(tokenizer._params.abbrev_types)


# Define the accesory function for preparing the feature of the classifier
def features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
 
    # init the stemmer
    stemmer = SnowballStemmer('english')
 
    # Pad the sequence with placeholders
    tokens = [('[START3]', '[START3]'),('[START2]', '[START2]'), ('[START1]', '[START1]')] +\
    list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]'), ('[END3]', '[END3]')]
    history = ['[START3]', '[START2]', '[START1]'] + list(history)
 
    # shift the index with 3, to accommodate the padding
    index += 3
 
    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    prev3word, prev3pos = tokens[index - 3]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    next3word, next3pos = tokens[index + 3]
    previob = history[index - 1]
    prevpreviob = history[index - 2]
    prev3iob = history[index - 3]
    contains_dash = '-' in word
    contains_dot = '.' in word
    allascii = all([True for c in word if c in string.ascii_lowercase])
 
    allcaps = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase
 
    prevallcaps = prevword == prevword.capitalize()
    prevcapitalized = prevword[0] in string.ascii_uppercase
 
    nextallcaps = prevword == prevword.capitalize()
    nextcapitalized = prevword[0] in string.ascii_uppercase
    
    is_math = lambda w:(w == '_inline_math_') or (w == '_display_math_')
    ismath = is_math(word)
    isprevmath = is_math(prevword)
    isprevprevmath = is_math(prevprevword)
 
    return {
        'word': word,
        'lemma': stemmer.stem(word),
                'pos': pos,
        'all-ascii': allascii,
 
        'next-word': nextword,
        'next-lemma': stemmer.stem(nextword),
        'next-pos': nextpos,
 
        'next-next-word': nextnextword,
        'nextnextpos': nextnextpos,
 
        'next3word': next3word,
        'next3pos': next3pos,
        
        'prev-word': prevword,
        'prev-lemma': stemmer.stem(prevword),
        'prev-pos': prevpos,
 
        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,
 
        'prev3word': prev3word,
        'prev3pos': prev3pos,
        
        'prev-iob': previob,
        
        'prev-prev-iob': prevpreviob,
 
        'contains-dash': contains_dash,
        'contains-dot': contains_dot,
 
        'all-caps': allcaps,
        'capitalized': capitalized,
 
        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,
 
        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized,
        
        'ismath': ismath,
        'isprevmath': isprevmath,
        'isprevprevmath': isprevprevmath,
    }

# Get the data and POS and NER tags for each definition (LONG TIME)
def_lst = []
for i in range(len(wiki)):
    try:
        title, section, defin_raw = wiki[i].split('-#-%-')
        defin_all = unwiki.loads(eval(defin_raw))
        for d in tokenizer.tokenize(defin_all):
            if title.lower().strip() in d.lower():
                pos_tokens = pos_tag(word_tokenize(d))
                def_ner = ner.bio_tag.bio_tagger(title.strip().split(), pos_tokens)
                other_ner = [((d[0],d[1]),d[2]) for d in def_ner]
                tmp_dict = {'title': title,
                           'section': section,
                           'defin': d,
                           'ner': other_ner}
                def_lst.append(tmp_dict)
    except ValueError:
        print('parsing error')
        
# The ChunkParserI has to be instantiated        
class NamedEntityChunker(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        assert isinstance(train_sents, Iterable)
 
        self.feature_detector = features
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=features,
            **kwargs)
    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)
 
        # Transform the result from [((w1, t1), iob1), ...] 
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]
 
        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)       
        
random.shuffle(def_lst)
training_samples = [d['ner'] for d in def_lst[:int(len(def_lst) * 0.9)]]
test_samples = [d['ner'] for d in def_lst[int(len(def_lst) * 0.9):]]
 
print("#training samples = %s" % len(training_samples) )   
print("#test samples = %s" % len(test_samples))            

#train the NER Chunking Classifier 
# %time chunker = NamedEntityChunker(training_samples)

# Evaluate the most common metrics on the test dataset
unpack = lambda l: [(tok, pos, ner) for ((tok, pos), ner) in l]
Tree_lst = [conlltags2tree(unpack(t)) for t in test_samples]
print(chunker.evaluate(Tree_lst))


def prepare_for_metrics(int_range, chunker_fn, data_set = test_samples, print_output=False):
    '''
    Accesory function for computing metrics
    `int_range` is an integer range
    NEEDS A TEST_SAMPLES VARIABLE CREATED WHEN SPLITTING THE 
    TRAINING AND TESTING DATA
    Returns two vectors ready to be used in the 
    metrics classification function
    '''
    if isinstance(int_range, int):
        int_range = [int_range]
    y_true = []
    y_pred = []
    for i in int_range:
        sample = data_set[i]
        sm = [s[0] for s in sample]
        y_true_tmp = [s[1] for s in sample]
        predicted = [v[2] for v in tree2conlltags(chunker_fn.parse(sm))]
        y_true += y_true_tmp
        y_pred += predicted
        if print_output:
            for k,s in enumerate(sm):
                print('{:15} {:>10}  {:>10}'.format(s[0], y_true_tmp[k], predicted[k]))
    return y_true, y_pred

# Prepare and print metrics for the normal metrics
OO = prepare_for_metrics(119, chunker, data_set=test_samples, print_output=True)
y_true, predicted = prepare_for_metrics(range(len(test_samples)), chunker)
print(metrics.classification_report(y_true, predicted))
# -

# An example of a user fed definition
chunked = chunker.parse(pos_tag(word_tokenize(Def[0])))
D =list(filter(lambda x: isinstance(x, nltk.tree.Tree), chunked))[0]
' '.join([d[0] for d in D])

art = px.DefinitionsXML('tests/latexmled_files/1501.06563.xml')
p_lst = [px.recutext_xml(p) for p in art.tag_list(tag='para')] 
p_vec = count_vect.transform(p_lst)
preds = clf.predict(p_vec)

for k,p in enumerate(p_lst):
    print(k,preds[k],p[:100])
    print('------')

chunk = tree2conlltags(chunker.parse(pos_tag(word_tokenize(p_lst[63]))))
for tok in chunk:
    print('{:15} {:>10} '.format(tok[0], tok[2]))

with open('../PickleJar/chunker.pickle', 'wb') as chunker_f:
    pickle.dump(chunker, chunker_f)

with open('data/vectorizer.pickle', 'wb') as token_f:
    pickle.dump(, token_f)
