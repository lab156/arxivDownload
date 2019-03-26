import unittest
import shutil
import parsing_xml as px
import nltk
from lxml import etree

class TestDefinitionsXML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xml1 = 1501.06563.xml

    def test_recutext_xml(self):


    def test_contain_words1(self):
        dtest = px.DefinitionsXML('tests/latexmled_files/math.0412433.xml')
        test_set = set(nltk.word_tokenize(dtest.get_def_text()[0].lower()))
        ss = {'.',
	     ';',
             ',',
	     'kirwan',
	     'let',
	     'we',
	     'be',
	     'codimension',
	     'components',
	     'divisorial',
	     'having',
	     'if',
	     'in',
	     'irreducible',
	     'is',
	     'locus',
	     'mild',
	     'of',
	     'one',
	     'other',
	     'part',
	     'resolution',
	     'say',
	     'shall',
	     'that',
	     'the',
	     'union',
	     'unstable',
	     'words',
             '_inline_math_', }
        self.assertSetEqual(ss, test_set)

    def test_contain_words2(self):
        dtest = px.DefinitionsXML('tests/latexmled_files/math.0402243.xml')
        test_set = set(nltk.word_tokenize(dtest.get_def_text()[0].lower()))
        ss = {'quotient',
                'singularités',
                'orbifolde'}
        self.assertTrue(ss.issubset(test_set))

    def test_exact_tokenize1(self):
        dtest = px.DefinitionsXML('tests/latexmled_files/math.0402243.xml')
        str1 = '''une orbifolde pure est un espace analytique complexe
        normal _inline_math_  n’ayant que des singularités\nquotient.'''
        str2 = dtest.get_def_text()[0].lower()

        self.assertEqual(nltk.word_tokenize(str1), nltk.word_tokenize(str2))

    def test_exact_tokenize2(self):
        dtest = px.DefinitionsXML('tests/latexmled_files/math.0412433.xml')
        str1 = '''let _inline_math_ \n             be\nthe divisorial part of the unstable locus of _inline_math_ \n            ; in other words,\n\n         _inline_math_     is the union of the irreducible components\nof \n            _inline_math_  having codimension one in _inline_math_ \n            . we shall say\nthat the kirwan resolution _inline_math_ \n             is mild if _inline_math_ \n       .'''

        str2 = dtest.get_def_text()[0].lower()
        self.assertEqual(nltk.word_tokenize(str1), nltk.word_tokenize(str2))

    def test_exact_tokenize3(self):
        dtest = px.DefinitionsXML('tests/latexmled_files/math.0407523.xml')
        list1 = ['a',
	     'coherent',
	     'system',
             '_inline_math_',
	     'is',
	     'injective',
	     'if',
	     'the',
	     'evaluation',
	     'morphism',
             '_inline_math_',
	     'is',
	     'injective',
	     'as',
	     'a',
	     'morphism',
	     'of',
	     'sheaves',
	     '.',
	     'moreover',
             '_inline_math_',
	     'is',
	     'torsion-free',
	     'if',
	     'it',
	     'is',
	     'injective',
	     'and',
	     'the',
	     'quotient',
	     'sheaf',
             '_inline_math_',
	     'is',
	     'torsion-free',
	     '.']
        list2 = dtest.get_def_text()[3].lower()
        self.assertEqual(list1, nltk.word_tokenize(list2))








