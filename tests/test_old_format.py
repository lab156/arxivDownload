import unittest
import os
import sys
import shutil
from process import Xtraction


class TestXtraction2(unittest.TestCase):
    def setUp(self):
        self.f_path = '/mnt/arXiv_src/src/arXiv_src_0701_002.tar'
        self.check_dir = os.path.join(os.path.curdir,'check_test')
        self.check_dir2 = os.path.join(os.path.curdir,'check_test2')
        print('Starting extraction of  %s         \r'%os.path.basename(self.f_path),end='\r')
        x = Xtraction(self.f_path)
        f_lst = x.filter_MSC('math.AG')
        for f in f_lst:
            print("\033[K",end='') 
            print('writing file %s               \r'%f,end='\r')
            x.extract_any(f , self.check_dir)
        print('successful extraction of  %s      '%os.path.basename(self.f_path))
        x.extract_tar(self.check_dir2, 'math.AG')

    def test_extract_any(self):
        list1 = ['math.0701750',
		 'math.0701670',
		 'math.0701246',
		 'math.0701046',
		 'math.0701036',
		 'math.0701409',
		 'math.0701532',
		 'math.0701662',
		 'math.0701002',
		 'math.0701374',
		 'math.0701091',
		 'math.0701559',
		 'math.0701566',
		 'math.0701105',
		 'math.0701903',
		 'math.0701213',
		 'math.0701319',
		 'math.0701734',
		 'math.0701669',
		 'math.0701507',
		 'math.0701671',
		 'math.0701402',
		 'math.0701538',
		 'math.0701683',
		 'math.0701782',
		 'math.0701885',
		 'math.0701475',
		 'math.0701388',
		 'math.0701895',
		 'math.0701590',
		 'math.0701823',
		 'math.0701459',
		 'math.0701667',
		 'math.0701874',
		 'math.0701641',
		 'math.0701257',
		 'math.0701663',
		 'math.0701053',
		 'math.0701680',
		 'math.0701336',
		 'math.0701889',
		 'math.0701511',
		 'math.0701074',
		 'math.0701312',
		 'math.0701522',
		 'math.0701115',
		 'math.0701720',
		 'math.0701407',
		 'math.0701839',
		 'math.0701194',
		 'math.0701472',
		 'math.0701466',
		 'math.0701936',
		 'math.0701894',
		 'math.0701479',
		 'math.0701138',
		 'math.0701672',
		 'math.0701255',
		 'math.0701596',
		 'math.0701311',
		 'math.0701456',
		 'math.0701521',
		 'math.0701376',
		 'math.0701546',
		 'math.0701353',
		 'math.0701867',
		 'math.0701487',
		 'math.0701502',
		 'math.0701423',
		 'math.0701297',
		 'math.0701877',
		 'math.0701620',
		 'math.0701110',
		 'math.0701315',
		 'math.0701406',
		 'math.0701642',
		 'math.0701597']


        set2 = set(['commentary.txt',  'definitions.sty',  'hyperplane.bbl',  'hyperplane.tex'])

        self.assertEqual(list1, os.listdir(self.check_dir))
        self.assertEqual(list1, os.listdir(self.check_dir2))

        self.assertEqual(set2, set(os.listdir(os.path.join(self.check_dir2, 'math.0701590'))))
        self.assertEqual(set2, set(os.listdir(os.path.join(self.check_dir, 'math.0701590'))))
        #with open(os.path.join(self.check_dir2,'math.0701590','1601.00103.tex'),'r') as tst_file:
#            self.assertEqual(list2, tst_file.readlines()[:10])
        with open(os.path.join(self.check_dir,'math.0701521','math.0701521.tex'),'r') as tst_file:
            with open(os.path.join(os.path.curdir, 'tests', 'mathfile.tex'), 'r') as gf:
                self.assertEqual(gf.readlines(), tst_file.readlines())


    def tearDown(self):
        print('removing %s'%self.check_dir)
        shutil.rmtree(self.check_dir)
        shutil.rmtree(self.check_dir2)
