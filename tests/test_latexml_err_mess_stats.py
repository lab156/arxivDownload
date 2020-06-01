import unittest
import latexml_err_mess_stats as err
import datetime as dt
import dateutil

#run with:
#export PYTHONPATH="./tests"; python3 -m unittest test_latexml_err_mess_stats

class TestParseConversion(unittest.TestCase):
    def test_parse_conversion1(self):
        test_str = '''\nConversion complete: 21 warnings; 101 errors; 1 fatal error; 8 undefined macros[\\xymatrixcolsep, \\save, \\txt, {IEEEeqnarray*}, \\centernot, \\restore, \\ar, \\xymatrix]; 3 missing files[centernot.sty, xy.sty, IEEEtrantools.sty].\n'''
        expect = (None, '21 warnings; ', '101 errors; ', '1 fatal error; ', '8 undefined macros[\\xymatrixcolsep, \\save, \\txt, {IEEEeqnarray*}, \\centernot, \\restore, \\ar, \\xymatrix]; ', '3 missing files[centernot.sty, xy.sty, IEEEtrantools.sty]')
        self.assertEqual(expect, err.parse_conversion(test_str))

    def test_parse_conversion2(self):
        test_str = '''\nConversion complete: 3 warnings; 1 error; 2 missing files[xy.sty, keyval.sty].\n'''
        expect = (None, '3 warnings; ', '1 error; ', None, None, '2 missing files[xy.sty, keyval.sty]')
        self.assertEqual(expect, err.parse_conversion(test_str))

    def test_parse_conversion3(self):
        test_str = '''\nConversion complete: No obvious problems.\n'''
        expect = ('No obvious problems.', None, None, None, None, None)
        self.assertEqual(expect, err.parse_conversion(test_str))

class TestParseLaTeXMLLog(unittest.TestCase):
    def test_time_space_init(self):
        with open('./tests/test_stats_files/test1/latexml_errors_mess.txt', 'rb') as log,\
                open('./tests/test_stats_files/test1/latexml_commentary.txt', 'rb') as comm:
            P = err.ParseLaTeXMLLog(log, comm, 'test1')
            self.assertEqual(24, P.time_secs)

    def test_commentary(self):
        with open('./tests/test_stats_files/test1/latexml_errors_mess.txt', 'rb') as log,\
                open('./tests/test_stats_files/test1/latexml_commentary.txt', 'rb') as comm:
            P = err.ParseLaTeXMLLog(log, comm, 'test1')
            self.assertEqual(len(P.commentary), 6)

    def test_time_space_init_dir(self):
        with open('./tests/test_stats_files/test1/latexml_errors_mess.txt', 'rb') as log,\
                open('./tests/test_stats_files/test1/latexml_commentary.txt', 'rb') as comm:
            P = err.ParseLaTeXMLLog(log, comm, 'test1')
            self.assertEqual(24, P.time_secs)

    def test_commentary_dir(self):
        with open('./tests/test_stats_files/test1/latexml_errors_mess.txt', 'rb') as log,\
                open('./tests/test_stats_files/test1/latexml_commentary.txt', 'rb') as comm:
            P = err.ParseLaTeXMLLog(log, comm, 'test1')
        self.assertEqual(len(P.commentary), 6)

    def test_init_parsing(self):
        with open('./tests/test_stats_files/test1/latexml_errors_mess.txt', 'rb') as log,\
                open('./tests/test_stats_files/test1/latexml_commentary.txt', 'rb') as comm:
            P = err.ParseLaTeXMLLog(log, comm, 'test1')
        self.assertEqual(P.warnings, 2)
        self.assertEqual(P.errors, 0)
        self.assertEqual(P.fatal_errors, 0)
        self.assertEqual(P.fatal_errors, 0)
        self.assertEqual(P.undefined_macros, 0)
        self.assertEqual(P.missing_files, 2)
        self.assertEqual(P.no_prob, None)

    def test_finished(self):
        with open('./tests/test_stats_files/test1/latexml_errors_mess.txt', 'rb') as log,\
                open('./tests/test_stats_files/test1/latexml_commentary.txt', 'rb') as comm:
            P = err.ParseLaTeXMLLog(log, comm, 'test1')
        self.assertEqual(P.finished(), 1200)

    def test_init_flags(self):
        with open('./tests/test_stats_files/test1/latexml_errors_mess.txt', 'rb') as log,\
                open('./tests/test_stats_files/test1/latexml_commentary.txt', 'rb') as comm:
            P = err.ParseLaTeXMLLog(log, comm, 'test1')
        self.assertEqual(P.result, err.Result.SUCC)
        self.assertTrue(not(P.result in err.Result.FAIL))
        with open('./tests/test_stats_files/test2/latexml_errors_mess.txt', 'rb') as log,\
                open('./tests/test_stats_files/test2/latexml_commentary.txt', 'rb') as comm:
            P = err.ParseLaTeXMLLog(log, comm, 'test1')
        self.assertEqual(P.result, err.Result.FATAL)
        self.assertTrue(P.result in err.Result.FAIL)
        with open('./tests/test_stats_files/test3/latexml_errors_mess.txt', 'rb') as log,\
                open('./tests/test_stats_files/test3/latexml_commentary.txt', 'rb') as comm:
            P = err.ParseLaTeXMLLog(log, comm, 'test1')
        self.assertEqual(P.result, err.Result.TIMED|err.Result.FATAL)
        self.assertTrue(P.result in err.Result.FAIL)
        with open('./tests/test_stats_files/test4/latexml_errors_mess.txt', 'rb') as log,\
                open('./tests/test_stats_files/test4/latexml_commentary.txt', 'rb') as comm:
            P = err.ParseLaTeXMLLog(log, comm, 'test1')
        self.assertEqual(P.result, err.Result.TIMED|err.Result.FATAL)
