import unittest
import sys
sys.path.append("../")
from definiendum import DefiniendumExtract

#From the tests directory run:
#python -m unittest test_definiendum.py

class TestDefiniendum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d1 = DefiniendumExtract(
                'https://en.wikipedia.org/wiki/Submanifold')
        cls.d2 = DefiniendumExtract(
                'https://groupprops.subwiki.org/wiki/Trivial_group')
        cls.d3 = DefiniendumExtract(
                'https://stacks.math.columbia.edu/tag/03NB')
        cls.deadend = DefiniendumExtract(
 'https://groupprops.subwiki.org/wiki/Understanding_the_definition_of_a_group')
        cls.deadend2 = DefiniendumExtract(
            'https://groupprops.subwiki.org/wiki/Verifying_the_group_axioms')
        cls.trailing = DefiniendumExtract(
                'https://en.wikipedia.org/wiki/Function_(mathematics)')

    def test_style_detection(self):
        self.assertEqual(self.d1.style, 'wikipedia')
        self.assertEqual(self.d2.style, 'subwiki')
        self.assertEqual(self.d3.style, 'stacks_proj')

    def test_title(self):
        self.assertEqual(self.d1.title(), 'submanifold')
        self.assertEqual(self.d2.title(), 'trivial group')
        self.assertEqual(self.d3.title(), 'presheaves')

    def test_defin_section(self):
        self.assertRegex(self.d1.defin_section().text, '^.*definition.*$')
        self.assertRegex(self.d2.defin_section().text, '^.*Definition.*$')
        self.assertRegex(self.d3.defin_section().text, '\\nDefinition.*$')

    def test_next(self):
        self.assertEqual(len(self.d1.next()), 1)
        self.assertEqual(self.d3.next()[0],
               'https://stacks.math.columbia.edu/tag/03NF')
        #Check if num_links is too large
        self.assertGreater(len(self.d1.next(num_links=200)),10)
        #Check if actually the size is equal to num_links
        self.assertEqual(len(self.d2.next(num_links=10)),10)

    def test_dead_end(self):
        self.assertIsNone(self.deadend.def_pair_or_none())
        self.assertIsNone(self.deadend2.def_pair_or_none())

    def test_stripping_trailing_label(self):
        self.assertEqual(self.trailing.title(), 'function')


