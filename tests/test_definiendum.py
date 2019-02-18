import unittest
from definiendum import DefiniendumExtract

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
        self.assertSetEqual(set(self.d1.next()),
                set(['https://en.wikipedia.org/wiki/Differentiable_manifold',
                     'https://en.wikipedia.org/wiki/Differentiability_class']))
        self.assertEqual(self.d3.next()[0],
               'https://stacks.math.columbia.edu/tag/03NF')
        #Check if num_links is too large
        self.assertEqual(len(self.d1.next(num_links=200)),38)
        #Check if actually the size is equal to num_links
        self.assertEqual(len(self.d2.next(num_links=10)),10)

    def test_dead_end(self):
        self.assertIsNone(self.deadend.def_pair_or_none())
