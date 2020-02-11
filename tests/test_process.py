import unittest
import os
import sys
import shutil
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from process import Xtraction, sliced_article_query
import databases.create_db_define_models as cre

class TestXtraction1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f_path = 'tests/minitest.tar'
        cls.f_path2 = 'tests/minitest2.tar'
        cls.f_path3 = 'tests/minitest3.tar'   # sample from tarfile 1403_001
        cls.check_dir = os.path.join(os.path.curdir,'check_test')
        cls.check_dir2 = os.path.join(os.path.curdir,'check_test2')
        cls.check_dir3 = os.path.join(os.path.curdir,'check_test3')
        cls.check_dir4 = os.path.join(os.path.curdir,'check_test4')
        cls.x = Xtraction(cls.f_path)
        cls.xdb = Xtraction(cls.f_path, db='sqlite:///tests/test.db')
        cls.x.extract_tar(cls.check_dir)
        cls.x2 = Xtraction(cls.f_path2)
        cls.xdb2 = Xtraction(cls.f_path2, db='sqlite:///tests/test.db')
        cls.x2.extract_tar(cls.check_dir2, 'all')
        cls.x3 = Xtraction(cls.f_path3)

    def test_save_articles_to_db(self):
        self.x3.save_articles_to_db('sqlite:///tests/test.db')
        x3db = Xtraction(self.f_path3, db='sqlite:///tests/test.db')
        self.assertEqual(['1403/1403.0035.pdf',
                 '1403/1403.0040.gz',
                 '1403/1403.0038.gz',
                 '1403/1403.0039.gz',
                 '1403/1403.0037.gz',
                 '1403/1403.0036.pdf'],
                 x3db.art_lst,
                 msg='this means that the save_articles_to_db did no store the files correctly')

    def test_sliced_article_query_slices(self):
        correct_lst = list(map(self.x.tar2api, self.x.art_lst))
        q_lst = sliced_article_query(correct_lst, slice_length=3)
        q_lst2 = sliced_article_query(correct_lst, slice_length=5)
        self.assertEqual(9, len(q_lst))
        self.assertEqual(9, len(q_lst2))

    def test_extract_article_by_name(self):


    def test_metadata_on_all_articles(self):
        for q in self.x.query_results:
            self.assertIn('id', q.keys())
        for q in self.x2.query_results:
            self.assertIn('id', q.keys())

    def test_info_on_every_article(self):
        self.assertEqual(len(self.x.query_results), len(self.x.art_lst), msg="some queries are missing")
        self.assertEqual(len(self.x2.query_results), len(self.x2.art_lst), msg="some queries are missing")

    def test_extract_tar_list(self):
        list1 = ['math.0303001',
                 'math.0303002',
                 'math.0303003',
                 'math.0303004',
                 'math.0303005',
                 'math.0303006',
                 'math.0303007',
                 'math.0303008',
                 'math.0303009', ]
        self.assertSetEqual(set(list1), set(os.listdir(self.check_dir)))

    def test_filter_arxiv_meta_api(self):
        self.assertEqual(len(self.x.art_lst), len(self.x.filter_arxiv_meta('math')))
        self.assertEqual(set(['0303/math0303004.gz', '0303/math0303008.gz', '0303/math0303006.gz']),
                set(self.x.filter_arxiv_meta('math.AP')))
        self.assertEqual(4, len(self.x2.filter_arxiv_meta('physics')))
        self.assertEqual(2, len(self.x2.filter_arxiv_meta('cs.NA', 'cs.DS')))

    def test_filter_arxiv_meta_db(self):
        self.assertEqual(len(self.xdb.art_lst), len(self.xdb.filter_arxiv_meta('math')))
        self.assertEqual(set(['0303/math0303004.gz', '0303/math0303008.gz', '0303/math0303006.gz']),
                set(self.xdb.filter_arxiv_meta('math.AP')))
        self.assertEqual(4, len(self.xdb2.filter_arxiv_meta('physics')))
        self.assertEqual(2, len(self.xdb2.filter_arxiv_meta('cs.NA', 'cs.DS')))

    def test_extract_tar_list2(self):
        list1 = ['1804.01586',
                 '1804.01592',
                 '1804.01583',
                 '1804.01585',
                 '1804.01588',
                 '1804.01589',
                 '1804.01587',
                 '1804.01584',
                 '1804.01591',
                 '1804.01590',
                 '1804.01593',]
        self.assertSetEqual(set(list1), set(os.listdir(self.check_dir2)))

    def test_extract_any_identify_pdf(self):
        with open(os.path.join(self.check_dir2,'1804.01587','commentary.txt'),'r') as tst_file:
            self.assertTrue('pdf file' in tst_file.read())
        with open(os.path.join(self.check_dir,'math.0303009','commentary.txt'),'r') as tst_file:
            self.assertTrue('pdf file' in tst_file.read())

    def test_extract_any_identify_cry_files(self):
        with open(os.path.join(self.check_dir,'math.0303004','commentary.txt'),'r') as tst_file:
            self.assertTrue('.cry file' in tst_file.read())
        with open(os.path.join(self.check_dir,'math.0303006','commentary.txt'),'r') as tst_file:
            self.assertTrue('.cry file' in tst_file.read())
        with open(os.path.join(self.check_dir,'math.0303008','commentary.txt'),'r') as tst_file:
            self.assertTrue('.cry file' in tst_file.read())


    def test_extract_text(self):
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

        with open(os.path.join(self.check_dir, 'math.0303007','1july05.tex'),'r') as tst_file:
            self.assertListEqual(expect_lst, tst_file.readlines()[:30])

    def test_extract_text2(self):
        expect_lst = ['% Version:\n',
    '\\newcommand{\\version}{May 30, 2018}\n',
    '\\documentclass[letterpaper,oneside,11pt,onecolumn,pdftex]{article}\n',
    '\\pdfoutput=1\n',
    '%\n',
    '% included libraries\n',
    '\\usepackage[bindingoffset=0.0cm,textheight=22.6cm,hdivide={*,16.0cm,*}, vdivide={*,22.6cm,*}]{geometry}\n',
    '\\usepackage{amsmath,amsfonts,amssymb}\n',
    '\\usepackage{mathtools}\n',
    '\\usepackage[pdftex]{graphicx}\n',
    '\\usepackage[margin=1cm,font=small,labelfont=bf]{caption}\n',
    '\\usepackage[T1]{fontenc}\n',
    '\\usepackage[utf8]{inputenc}\n',
    '\\usepackage{fouriernc}\n',
    '\\newcommand{\\id}{\\mathbb{1}}\n',]

        with open(os.path.join(self.check_dir2, '1804.01586','disloc-drag-semiiso-v2.tex'),'r') as tst_file:
            tst_file_lines = tst_file.readlines()
            self.assertListEqual(expect_lst, tst_file_lines[:15])
            self.assertEqual(len(tst_file_lines), 1050)


    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.check_dir)
        shutil.rmtree(cls.check_dir2)
        #shutil.rmtree(cls.check_dir3)

        eng = sa.create_engine('sqlite:///tests/test.db', echo=True)
        eng.connect()
        SMaker = sessionmaker(bind=eng)
        sess = SMaker()
        sess.query(cre.Article).filter(cre.Article.tarfile_id == 5).delete(synchronize_session='fetch')
        sess.commit()
