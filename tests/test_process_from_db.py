import unittest
import process_from_db as from_db

class TestFromDB(unittest.TestCase):
    def test_extract_from_database(self):
        set1 = set(['1601.00133',
                 '1601.00089',
                 '1601.00103',
                 '1601.00502',
                 '1601.00340',
                 '1601.00302'])
        file_lst = from_db.query_files('src/arXiv_src_1601_001.tar',
                'math.AG', 'arxiv1.db')
        trimmed_set = set([from_db.trim_id(a[0]) for a in file_lst])
        self.assertEqual(set1, trimmed_set)

    def test_old_format_from_db(self):
        set2 = set(['math/9803001',
            'math/9803007',
            'math/9803010',
            'math/9803013',
            'math/9803017',
            'math/9803026',
            'math/9803028',
            'math/9803029',
            'math/9803033',
            'math/9803036',
            'math/9803039',
            'math/9803040',
            'math/9803041',
            'math/9803047',
            'math/9803048',
            'math/9803053',
            'math/9803054',
            'math/9803065',
            'math/9803066',
            'math/9803069',
            'math/9803071',
            'math/9803072',
            'math/9803076',
            'math/9803078',
            'math/9803091',
            'math/9803094',
            'math/9803096',
            'math/9803103',
            'math/9803107',
            'math/9803108',
            'math/9803111',
            'math/9803112',
            'math/9803113',
            'math/9803119',
            'math/9803120',
            'math/9803121',
            'math/9803124',
            'math/9803126',
            'math/9803131',
            'math/9803141',
            'math/9803143',
            'math/9803144',
            'math/9803145',
            'math/9803150',
            'math/9803152'])
        file_lst = from_db.query_files('src/arXiv_src_9803_001.tar',
                'math.AG', 'arxiv1.db')
        trimmed_set = set([from_db.trim_id(a[0]) for a in file_lst])
        self.assertEqual(set2, trimmed_set)
