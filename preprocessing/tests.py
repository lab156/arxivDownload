import unittest
import prepro as pre

class TestClassCommandCleaner(unittest.TestCase):
    def setUp(self):
        with open('../tests/tex_files/short_xymatrix_example.tex')\
                as xymatrix_file:
            self.short_example = xymatrix_file.read()

    def test_show_matches(self):
        cc = pre.CommandCleaner('xymatrix')
        result = cc.show_matches(self.short_example)[0][0]
        should_be = '(\\Sh(\\mathcal C ), \\underline B ) \\ar[r]_i \\ar[d]_\\pi & (\\Sh(\\mathcal C ), \\mathcal O ) \\\\ (\\Sh(*), B)'
        self.assertEqual(result, should_be)
