import unittest
import shutil
import parsing_xml as px
import nltk

class TestDefinitionsXML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = {'ltx': 'http://dlmf.nist.gov/LaTeXML'}
        cls.xml1 = px.DefinitionsXML('tests/latexmled_files/1501.06563.xml')
        cls.xml_lst1 = cls.xml1.exml.findall('.//ltx:p', namespaces=cls.ns)
        cls.html1 = px.DefinitionsXML('tests/latexmled_files/1501.06563.html')
        cls.html2 = px.DefinitionsXML('tests/latexmled_files/1501.06563_shortened.html')
        cls.html_lst1 = cls.html1.exml.findall('.//p', namespaces=cls.ns)


    def test_recutext_xml(self):
        expect1 = 'For the remaining properties we state we shall assume that _inline_math_ or _inline_math_.'
        expect2 = '''Let _inline_math_ be a set of elements of _inline_math_. Recall that an _inline_math_-invariant CAD of _inline_math_ _citation_ is a partitioning of _inline_math_ into connected subsets called cells compatible with the zeros of the elements of _inline_math_. The output of a CAD algorithm applied to _inline_math_ is a description of an _inline_math_-invariant CAD _inline_math_ of _inline_math_. That is, _inline_math_ is a decomposition of _inline_math_ determined by the roots of the elements of _inline_math_ over the cells of some cylindrical algebraic decomposition _inline_math_ of _inline_math_; each element of _inline_math_ is sign-invariant throughout every cell of _inline_math_.'''
        self.assertEqual(expect1, px.recutext_xml(self.xml_lst1[19]))
        self.assertEqual(expect2, px.recutext_xml(self.xml_lst1[32]))

    def test_same_xml_and_html(self):
        self.assertEqual(px.recutext_xml(self.xml_lst1[34]),
                            px.recutext_html(self.html_lst1[34]))
        self.assertEqual(px.recutext_xml(self.xml_lst1[44]),
                            px.recutext_html(self.html_lst1[44]))
        self.assertEqual(px.recutext_xml(self.xml_lst1[4]),
                            px.recutext_html(self.html_lst1[4]))

    def test_recutext_xml_html_similarity(self):
        #Get the para tags
        para_xml_lst = list(map(px.recutext_xml, self.xml1.exml.findall('.//ltx:para', namespaces=self.ns)))
        para_html_lst = list(map(px.recutext_html,
            self.html1.exml.xpath(".//div[contains(@class, 'ltx_para')]" )))
        self.assertEqual(para_xml_lst, para_html_lst)

    def test_defin_finding(self):
        self.assertEqual(3, len(self.xml1.get_def_text()))
        self.assertEqual(3, len(self.html1.get_def_text()))

    def test_no_repeat_in_sample(self):
        test_dict = self.html2.get_def_sample_text_with()
        self.assertEqual(3, len(test_dict['real']))
        self.assertEqual(0, len(test_dict['nondef']))

    def test_defin_xml_html_equal(self):
        self.assertEqual(self.xml1.get_def_text(), self.html1.get_def_text())

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


