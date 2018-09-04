import unittest
import shutil
from parsing_xml import DefinitionsXML
import nltk

class TestDefinitionsXML(unittest.TestCase):
    def test_contain_words1(self):
        dtest = DefinitionsXML('tests/latexmled_files/math.0412433.xml')
        test_set = set(nltk.word_tokenize(dtest.get_def_text()[0]))
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
	     'words'}
        self.assertSetEqual(ss, test_set)

    def test_contain_words2(self):
        dtest = DefinitionsXML('tests/latexmled_files/math.0402243.xml')
        test_set = set(nltk.word_tokenize(dtest.get_def_text()[0]))
        ss = {'quotient',
                'singularités',
                'orbifolde'}
        self.assertTrue(ss.issubset(test_set))

    def test_exact_tokenize1(self):
        dtest = DefinitionsXML('tests/latexmled_files/math.0402243.xml')
        str1 = '''une orbifolde pure est un espace analytique complexe
        normal  n’ayant que des singularités\nquotient.'''
        str2 = dtest.get_def_text()[0]

        self.assertEqual(nltk.word_tokenize(str1), nltk.word_tokenize(str2))

    def test_exact_tokenize2(self):
        dtest = DefinitionsXML('tests/latexmled_files/math.0412433.xml')
        str1 = '''let \n             be\nthe divisorial part of the unstable locus of \n            ; in other words,\n\n             is the union of the irreducible components\nof \n             having codimension one in \n            . we shall say\nthat the kirwan resolution \n             is mild if \n       .'''

        str2 = dtest.get_def_text()[0]
        self.assertEqual(nltk.word_tokenize(str1), nltk.word_tokenize(str2))





                
