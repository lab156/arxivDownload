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

class MockTokenizer():
    def __init__(self):
        pass
    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            length = len(text.split)
            return { 'input_ids': np.random.randint(3e5, size=length),
                    'token_type_ids': np.zeros(length),
                    'attention_mask': np.ones(length),
                    }
        else:
            ii = [] # input_ids list
            tti = [] # token_type_ids list
            am = []  # attention mask_ids
            for t in text:
                length = len(t.split())
                ii.append(np.random.randint(3e5, size=length))
                tti.append(np.zeros(length))
                am.append(np.ones(length))
            return { 'input_ids': ii,
                    'token_type_ids': tti, 
                    'attention_mask': am,
                    }
    def pad(self, x, **kwargs):
        return x[:10]


class TestDefiniendumInit(unittest.TestCase):
    def test_definition_index_is_article_index1(self):
        clf = FarseClf()
        vzer = FarseVectorizer()
        bio = FarseBio()
        for fname, tfobj in peep.tar_iter('few_actual_articles.tar.gz', '.xml'):
            parsing_fobj = px.DefinitionsXML(tfobj)
            try:
                Def = X.Definiendum(parsing_fobj, clf, bio, vzer, None, min_words=240)
            except ValueError:
                pass
        idx = Def.root.findall('definition')[0].attrib['index']
        idx = int(idx)
        # This value should not depend on `min_words`
        self.assertEqual(idx,180)

    def test_definition_index_is_article_index2(self):
        clf = FarseClf()
        vzer = FarseVectorizer()
        bio = FarseBio()
        for fname, tfobj in peep.tar_iter('few_actual_articles.tar.gz', '.xml'):
            parsing_fobj = px.DefinitionsXML(tfobj)
            try:
                Def = X.Definiendum(parsing_fobj, clf, bio, vzer, None, min_words=20)
            except ValueError:
                pass
        idx = Def.root.attrib['num']
        idx = int(idx)
        # This value should not depend on `min_words`
        self.assertEqual(idx,272)

    def test_prob_predict_with_threshold(self):
        clf = RandClf()
        vzer = FarseVectorizer()
        bio = None
        for fname, tfobj in peep.tar_iter('few_actual_articles.tar.gz', '.xml'):
            parsing_fobj = px.DefinitionsXML(tfobj)
            try:
                Def = X.Definiendum(parsing_fobj, clf, bio, vzer, None, min_words=40,
                        thresh=0.96)
            except ValueError:
                pass
        # This value should not depend on `min_words`
        self.assertEqual(len(Def.root), 4)

    def test_vectorizer_raises_valueerror(self):
        clf = RandClf()
        vzer = FarseVectorizer()
        xml = peep.tar('some_empty_articles.tar.gz', 1)[1]
        with self.assertRaises(ValueError):
            X.Definiendum(xml, clf, None, vzer, None, min_words=40)

    def test_incompatible_arguments(self):
        clf = RandClf()
        vzer = FarseVectorizer()
        tzer = MockTokenizer()
        xml = peep.tar('few_actual_articles.tar.gz', 1)[1]
        with self.assertRaises(AssertionError):
            X.Definiendum(xml, clf, None, vzer, tzer, min_words=40)
        with self.assertRaises(AssertionError):
            X.Definiendum(xml, clf, None, None, tzer, thresh=0.90, min_words=40)


