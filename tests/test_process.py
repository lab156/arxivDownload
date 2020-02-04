import unittest
import os
import sys
import shutil
from process import Xtraction

class TestXtraction1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f_path = 'tests/minitest.tar'
        cls.f_path2 = 'tests/minitest2.tar'
        cls.check_dir = os.path.join(os.path.curdir,'check_test')
        cls.check_dir2 = os.path.join(os.path.curdir,'check_test2')
        cls.check_dir3 = os.path.join(os.path.curdir,'check_test3')
        print('Starting extraction of  %s         \r'%os.path.basename(cls.f_path),end='\r')
        x = Xtraction(cls.f_path)
        x2 = Xtraction(cls.f_path2)
        f_lst = x.filter_MSC('math.AP')
        for f in f_lst:
            print("\033[K",end='')
            print('writing file %s               \r'%f,end='\r')
            x.extract_any(f , cls.check_dir)
        print('successful extraction of  %s      '%os.path.basename(cls.f_path))
        x.extract_any(x.filter_MSC('math.DS')[0], cls.check_dir2)
        x.extract_any(x.filter_MSC('math.GM')[0], cls.check_dir2)
        x2.extract_any(x2.filter_MSC('cs.NA')[0], cls.check_dir3)
        x2.extract_any(x2.filter_MSC('physics.optics')[0], cls.check_dir3)
        x2.extract_any(x2.filter_MSC('cond-mat.mtrl-sci')[0], cls.check_dir3)

    def test_extract_any(self):
        list1 = [ 'math.0303004',
                  'math.0303006',
                  'math.0303008',]
        list2 = ['tar_file: minitest.tar\n',
            'encoding detected: None\n',
            'decode_message: Unknown encoding: None in file: 0303/math0303004 decoding with utf8\n',
            'decode_failed: Possible encrypted file (.cry) found.\n',]

        self.assertSetEqual(set(list1), set(os.listdir(self.check_dir)))
        with open(os.path.join(self.check_dir,'math.0303004','commentary.txt'),'r') as tst_file:
            self.assertEqual(list2, tst_file.readlines())

    def test_extract_any2(self):
        expect_lst =  ['% 1 july 05\n',
                '%*********************************************************\n',
                '% This paper has  9 encapsulated postscript figure files:\n',
                '% fig1.ps, fig2.ps, fig3.ps, fig4.ps, fig5.ps, fig6.ps.\n',
                '% fig7.ps, fig8.ps, fig9.ps \n',
                '%*********************************************************\n',
                '\\documentclass[11pt]{article}\n',
                '%\\usepackage{showlabels}\n',
                '\\usepackage{amssymb,psfig,latexsym}\n',
                '%\\renewcommand{\\labelenumi}{(\\arabic{enumi})}\n',
                '\\renewcommand{\\thefigure}{\\thesection.\\arabic{figure}}\n',
                '\\renewcommand{\\thetable}{\\thesection.\\arabic{table}}\n',
                  '\n',
                '\\newtheorem{theorem}{Theorem}[section]\n',
                '\\newtheorem{lemma}{Lemma}[section]\n',
                '\\newtheorem{exam}{Example}[section]\n',
                '\\newtheorem{prop}{Proposition}[section]\n',
                  '\n',
                '\\setlength{\\textheight}{8.5in}\n',
                '\\setlength{\\textwidth}{6.25in}\n',
                '\\setlength{\\oddsidemargin}{.15in}\n',
                '\\setlength{\\topmargin}{-.15in}\n',
                '\\setlength{\\headsep}{.5in}\n',
                '\\newcommand{\\bsq}{\\vrule height .9ex width .8ex depth -.1ex}\n',
                  '\n',
                '\\newcommand{\\ztzs}{\\zeta_T (z,s)}\n',
                '\\newcommand{\\req}[1]{(\\ref{#1})}\n',
                    '\\newcommand{\\lf}{\\lfloor}\n',
                    '\\newcommand{\\rf}{\\rfloor}\n',
                    '\\newcommand{\\dd}{{, \\ldots ,}}\n',]

        with open(os.path.join(self.check_dir2,'math.0303007','1july05.tex'),'r') as tst_file:
            self.assertListEqual(expect_lst, tst_file.readlines()[:30])

    def test_extract_any_identify_pdf(self):
        with open(os.path.join(self.check_dir2,'math.0303009','commentary.txt'),'r') as tst_file:
            self.assertTrue('pdf only' in tst_file.read())
        with open(os.path.join(self.check_dir3,'1804.01587','commentary.txt'),'r') as tst_file:
            self.assertTrue('pdf only' in tst_file.read())

    @classmethod
    def tearDownClass(cls):
        print('removing %s'%cls.check_dir)
        shutil.rmtree(cls.check_dir)
        shutil.rmtree(cls.check_dir2)
        shutil.rmtree(cls.check_dir3)
