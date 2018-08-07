import unittest
import os
import sys
import shutil
from process import Xtraction

class TestXtraction(unittest.TestCase):
    def setUp(self):
        self.f_path = '/mnt/arXiv_src/src/arXiv_src_1601_001.tar'
        self.check_dir = os.path.join(os.path.curdir,'check_test')
        self.check_dir2 = os.path.join(os.path.curdir,'check_test2')
        print('Starting extraction of  %s         \r'%os.path.basename(self.f_path),end='\r')
        x = Xtraction(self.f_path)
        f_lst = x.filter_MSC('math.AG')
        for f in f_lst:
            print("\033[K",end='') 
            print('writing file %s               \r'%f,end='\r')
            x.extract_any(f, self.check_dir)
        print('successful extraction of  %s      '%os.path.basename(self.f_path))
        x.extract_tar(self.check_dir2, 'math.AG')

    def test_extract_any(self):
        list1 = ['1601.00133',
                 '1601.00089',
                 '1601.00103',
                 '1601.00502',
                 '1601.00340',
                 '1601.00302']
        self.assertEqual(list1, os.listdir(self.check_dir))
        self.assertEqual(list1, os.listdir(self.check_dir2))


    def tearDown(self):
        print('removing %s'%self.check_dir)
        shutil.rmtree(self.check_dir)
        shutil.rmtree(self.check_dir2)
        
