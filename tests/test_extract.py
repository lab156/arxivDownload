import unittest
import shutil
import parsing_xml as px
import nltk
import peep_tar as peep
import extract as X
from lxml import etree
import numpy as np

# From the tests directory, run:
#PYTHONPATH=".." python3 -m unittest -v test_extract.py

class FarseClf():
    def __init__(self):
        pass
    def predict(self, lst, **kwargs):
        return [True for _ in lst]

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

class TestDefiniendumInit(unittest.TestCase):
    def test_definition_index_is_article_index1(self):
        clf = FarseClf()
        vzer = FarseVectorizer()
        bio = FarseBio()
        for fname, tfobj in peep.tar_iter('five_actual_articles.tar.gz', '.xml'):
            parsing_fobj = px.DefinitionsXML(tfobj)
            Def = X.Definiendum(parsing_fobj, clf, bio, vzer, None, min_words=240)
        idx = Def.root.findall('definition')[0].attrib['index']
        idx = int(idx)
        # This value should not depend on `min_words`
        self.assertEqual(idx,180)

    def test_definition_index_is_article_index2(self):
        clf = FarseClf()
        vzer = FarseVectorizer()
        bio = FarseBio()
        for fname, tfobj in peep.tar_iter('five_actual_articles.tar.gz', '.xml'):
            parsing_fobj = px.DefinitionsXML(tfobj)
            Def = X.Definiendum(parsing_fobj, clf, bio, vzer, None, min_words=200)
        idx = Def.root.attrib['num']
        idx = int(idx)
        # This value should not depend on `min_words`
        self.assertEqual(idx,272)

    def test_prob_predict_with_threshold(self):
        clf = RandClf()
        vzer = FarseVectorizer()
        bio = None
        for fname, tfobj in peep.tar_iter('five_actual_articles.tar.gz', '.xml'):
            parsing_fobj = px.DefinitionsXML(tfobj)
            Def = X.Definiendum(parsing_fobj, clf, bio, vzer, None, min_words=40,
                    thresh=0.96)
        # This value should not depend on `min_words`
        self.assertEqual(len(Def.root), 4)

