import unittest
import shutil
from parsing_xml import DefinitionsXML

class TestDefinitionsXML(unittest.TestCase):
    def test_contain_words1(self):
        dtest = DefinitionsXML('tests/latexmled_files/math.0412433.xml')
        test_set = set(dtest.get_def_text()[0].split())
        ss = {'.',
	     ';',
	     'Kirwan',
	     'Let',
	     'We',
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
	     'words,'}
        self.assertSetEqual(ss, test_set)

                
