import unittest
import prepro as pre

class TestClassCommandCleaner(unittest.TestCase):
    def setUp(self):
        with open('../tests/tex_files/short_xymatrix_example.tex')\
                as xymatrix_file:
            self.short_example = xymatrix_file.read()
        self.short_text = '''
$\\underline{B}$. We obtain the following diagram of morphisms of
ringed topoi
\\begin{equation}
\\label{equation-pi}
\\vcenter{
\\xymatrix{
(\\Sh(\\mathcal{C}), \\underline{B}) \\ar[r]_i \\ar[d]_\\pi &
(\\Sh(\\mathcal{C}), \\mathcal{O}) \\\\
(\\Sh(*), B)
}
}
\\end{equation}
The morphism $i$ is the identity on underlying topoi and
'''

    def test_show_matches(self):
        cc = pre.CommandCleaner('xymatrix')
        result = cc.show_matches(self.short_example)[0][0]
        should_be = '(\\Sh(\\mathcal C ), \\underline B ) \\ar[r]_i \\ar[d]_\\pi & (\\Sh(\\mathcal C ), \\mathcal O ) \\\\ (\\Sh(*), B)'
        self.assertEqual(result, should_be)

    def test_delete_matches(self):
        cc = pre.CommandCleaner('xymatrix')
        result = cc.del_matches(self.short_example)
        should_be = 'We endow $\\mathcal{C}$ with the chaotic topology\n(Sites, Example \\ref{sites-example-indiscrete}), i.e., we endow\n$\\mathcal{C}$ with the structure of a site where coverings are given by\nidentities so that all presheaves are sheaves.\nMoreover, we endow $\\mathcal{C}$ with two sheaves of rings. The first\nis the sheaf $\\mathcal{O}$ which sends to object $(P, \\alpha)$ to $P$.\nThen second is the constant sheaf $B$, which we will denote\n$\\underline{B}$. We obtain the following diagram of morphisms of\nringed topoi\n\\begin{equation}\n\\label{equation-pi}\n\\vcenter{\n \n}\n\\end{equation}\nThe morphism $i$ is the identity on underlying topoi and\n$i^\\sharp : \\mathcal{O} \\to \\underline{B}$ is the obvious map.\nThe map $\\pi$ is as in Cohomology on Sites, Example\n\\ref{sites-cohomology-example-category-to-point}.\nAn important role will be played in the following\nby the derived functors\n'
        self.assertEqual(result, should_be)

    def test_delete_multiple_matches(self):
        cc = pre.CommandCleaner('xymatrix', 'mathcal')
        result = cc.del_matches(self.short_example)
        should_be = 'We endow $ $ with the chaotic topology\n(Sites, Example \\ref{sites-example-indiscrete}), i.e., we endow\n$ $ with the structure of a site where coverings are given by\nidentities so that all presheaves are sheaves.\nMoreover, we endow $ $ with two sheaves of rings. The first\nis the sheaf $ $ which sends to object $(P, \\alpha)$ to $P$.\nThen second is the constant sheaf $B$, which we will denote\n$\\underline{B}$. We obtain the following diagram of morphisms of\nringed topoi\n\\begin{equation}\n\\label{equation-pi}\n\\vcenter{\n \n}\n\\end{equation}\nThe morphism $i$ is the identity on underlying topoi and\n$i^\\sharp :   \\to \\underline{B}$ is the obvious map.\nThe map $\\pi$ is as in Cohomology on Sites, Example\n\\ref{sites-cohomology-example-category-to-point}.\nAn important role will be played in the following\nby the derived functors\n'
        self.assertEqual(result, should_be)

    def test_delete_matches2(self):
        cc = pre.CommandCleaner('vcenter')

        result = cc.del_matches(self.short_text)
        print(result)
        should_be = '''
$\\underline{B}$. We obtain the following diagram of morphisms of
ringed topoi
\\begin{equation}
\\label{equation-pi}
 
\\end{equation}
The morphism $i$ is the identity on underlying topoi and
'''
        self.assertEqual(result, should_be)


