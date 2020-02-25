import unittest
import latexml_err_mess_stats as err
import datetime as dt
import dateutil


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
        P = err.ParseLaTeXMLLog('./tests/test_stats_files/latexml_errors_mess.txt')
        self.assertEqual(24, P.time_secs)

