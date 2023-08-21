import unittest
import ner.llm_utils as llu

# RUNS
# From inside the tests directory, 
#  PYTHONPATH=".." python3 -m unittest -v test_llm_utils.py

tok_lst = ['[CLS]',
         'Su',
         '##pp',
         '##ose',
         'that',
         '_',
         'in',
         '##line',
         '_',
         'math',
         '_',
         '.',
         '[SEP]',
         '[PAD]',
         '[PAD]',]

class TestLLMUtils(unittest.TestCase):
    def test_get_words_back(self):
        expected_result = ['[CLS]',
						 'Suppose',
						 'that',
						 '_',
						 'inline',
						 '_',
						 'math',
						 '_',
						 '.',
                         '[SEP]',
                         '[PAD]',
                         '[PAD]',]
        self.assertEqual(llu.get_words_back(tok_lst)[0], 
                expected_result)

    def test_get_words_back3(self):
        test_lst = [('a', 'O'),
                    ('b', 'O'),
                    ('b', 'O'),
                    ('b', 'O'),
                    ('b', 'B-DFNDUM'),
                    ('##b', 'O'),
                    ('##b', 'O'),
                    ('##b', 'I-DFNDUM'),
                    ('##b', 'I-DFNDUM'),
                    ('b', 'I-DFNDUM'),]
                    
        w, p = zip(*test_lst)
        out_w, out_p = llu.get_words_back(w, preds=p)
        expect_w, expect_p = (['a', 'b', 'b', 'b', 'bbbbb',    'b'],
                              ['O', 'O', 'O', 'O', 'B-DFNDUM', 'I-DFNDUM'])
        self.assertEqual(out_w, expect_w)
        self.assertEqual(out_p, expect_p)


    def test_join_math_tokens(self):
        input_lst = ['[CLS]',
						 'Suppose',
						 'that',
						 '_',
						 'inline',
						 '_',
						 'math',
						 '_',
						 '.',
                         '[SEP]',
                         '[PAD]',
                         '[PAD]',]
        expected_lst = ['[CLS]',
						 'Suppose',
						 'that',
						 '_inline_math_',
						 '.',
                         '[SEP]',
                         '[PAD]',
                         '[PAD]',]
        self.assertEqual(llu.join_math_tokens(input_lst)[0], 
                expected_lst)

    def test_get_words_back2(self):
        in_tokens = ['we',                    
            'call',                  
            'the',                   
            'sub',                   
            '##mani',                
            '##fold',                
            '_',                     
            'in',                    
            '##line',                
            '_',                     
            'math',                  
            '_',                     
            'a',                     
            'final',                 
            'con',                   
            '##stra',                
            '##int',                 
            'sub',                   
            '##mani',                
            '##fold',                
            'at',                    
            '_',                     
            'in',                    
            '##line',                
            '_',                     
            'math', ]                       

        expected_tokens = ['we',                    
            'call',                  
            'the',                   
            'submanifold',                
            '_',                     
            'inline',                
            '_',                     
            'math',                  
            '_',                     
            'a',                     
            'final',                 
            'constraint',                 
            'submanifold',                
            'at',                    
            '_',                     
            'inline',                
            '_',                     
            'math', ]                       

        in_preds = [  'O',
                 'O',
                'O',
                'O',
                   'O',
                   'O',
              'O',
               'O',
                   'O',
              'O',
                 'O',
              'O',
              'O',
                  'B-DFNDUM',
                'I-DFNDUM',
                   'I-DFNDUM',
                  'I-DFNDUM',
                'I-DFNDUM',
                   'I-DFNDUM',
                   'I-DFNDUM',
               'O',
              'O',
               'O',
                   'O',
              'O',
                 'O',]

        expected_preds = [  'O',
                 'O',
                   'O',
              'O',
               'O',
                   'O',
              'O',
                 'O',
              'O',
              'O',
                  'B-DFNDUM',
                'I-DFNDUM',
                'I-DFNDUM',
               'O',
              'O',
               'O',
              'O',
                 'O',]

        out_toks, out_preds = llu.get_words_back(in_tokens, preds = in_preds)

        self.assertEqual(expected_tokens, out_toks)
        self.assertEqual(expected_preds, out_preds)

    def test_get_words_back_and_join_math_together(self):
        in_data = [('[CLS]',                      'O'),
                ('A',                      'O'),
                ('mon',                      'O'),
                ('##oid',                      'O'),
                ('##al',                      'O'),
                ('category',                      'O'),
                ('is',                      'O'),
                ('symmetric',                      'B-DFNDUM'),
                ('mon',                      'I-DFNDUM'),
                ('##oid',                      'I-DFNDUM'),
                ('##al',                      'I-DFNDUM'),
                ('if',                      'O'),
                ('it',                      'O'),
                ('has',                      'O'),
                ('the',                      'O'),
                ('special',                      'O'),
                ('arrow',                      'O'),
                ('_',                      'O'),
                ('display',                      'O'),
                ('_',                      'O'),
                ('math',                      'O'),
                ('_',                      'O'),
                ('for',                      'O'),
                ('every',                      'O'),]
        expected_data = [ ('A',                      'O'),
                ('monoidal',                      'O'),
                ('category',                      'O'),
                ('is',                      'O'),
                ('symmetric',                      'B-DFNDUM'),
                ('monoidal',                      'I-DFNDUM'),
                ('if',                      'O'),
                ('it',                      'O'),
                ('has',                      'O'),
                ('the',                      'O'),
                ('special',                      'O'),
                ('arrow',                      'O'),
                ('_',                      'O'),
                ('display',                      'O'),
                ('_',                      'O'),
                ('math',                      'O'),
                ('_',                      'O'),
                ('for',                      'O'),
                ('every',                      'O'),]
        words, preds = llu.get_words_back(list(zip(*in_data))[0],
                list(zip(*in_data))[1],
                special_tokens=['[CLS]', '[PAD]'])
        ed1, ed2 = zip(*expected_data)
        self.assertEqual(list(ed1), words)
        self.assertEqual(list(ed2), preds)


        expected_data2 = [ ('A',                      'O'),
                ('monoidal',                      'O'),
                ('category',                      'O'),
                ('is',                      'O'),
                ('symmetric',                      'B-DFNDUM'),
                ('monoidal',                      'I-DFNDUM'),
                ('if',                      'O'),
                ('it',                      'O'),
                ('has',                      'O'),
                ('the',                      'O'),
                ('special',                      'O'),
                ('arrow',                      'O'),
                ('_display_math_',                      'O'),
                ('for',                      'O'),
                ('every',                      'O'),]
        ed1, ed2 = zip(*expected_data2)

        words2, preds2 = llu.join_math_tokens(words, preds=preds)
        self.assertEqual(list(ed1), words2)
        self.assertEqual(list(ed2), preds2)

    def test_get_words_back_and_join_math_together2(self):
        in_data = [('[CLS]',                      'O'),
                ('A',                      'O'),
                ('mon',                      'O'),
                ('##oid',                      'O'),
                ('##al',                      'O'),
                ('category',                      'O'),
                ('is',                      'O'),
                ('symmetric',                      'B-DFNDUM'),
                ('mon',                      'I-DFNDUM'),
                ('##oid',                      'O'),
                ('##al',                      'I-DFNDUM'),
                ('if',                      'O'),
                ('it',                      'O'),
                ('has',                      'O'),
                ('the',                      'O'),
                ('special',                      'O'),
                ('arrow',                      'O'),
                ('_',                      'O'),
                ('display',                      'O'),
                ('_',                      'O'),
                ('math',                      'O'),
                ('_',                      'O'),
                ('for',                      'O'),
                ('every',                      'O'),]
        expected_data = [ ('A',                      'O'),
                ('monoidal',                      'O'),
                ('category',                      'O'),
                ('is',                      'O'),
                ('symmetric',                      'B-DFNDUM'),
                ('monoidal',                      'I-DFNDUM'),
                ('if',                      'O'),
                ('it',                      'O'),
                ('has',                      'O'),
                ('the',                      'O'),
                ('special',                      'O'),
                ('arrow',                      'O'),
                ('_',                      'O'),
                ('display',                      'O'),
                ('_',                      'O'),
                ('math',                      'O'),
                ('_',                      'O'),
                ('for',                      'O'),
                ('every',                      'O'),]
        words, preds = llu.get_words_back(list(zip(*in_data))[0],
                list(zip(*in_data))[1],
                special_tokens=['[CLS]', '[PAD]'])
        ed1, ed2 = zip(*expected_data)
        self.assertEqual(list(ed1), words)
        self.assertEqual(list(ed2), preds)


        expected_data2 = [ ('A',                      'O'),
                ('monoidal',                      'O'),
                ('category',                      'O'),
                ('is',                      'O'),
                ('symmetric',                      'B-DFNDUM'),
                ('monoidal',                      'I-DFNDUM'),
                ('if',                      'O'),
                ('it',                      'O'),
                ('has',                      'O'),
                ('the',                      'O'),
                ('special',                      'O'),
                ('arrow',                      'O'),
                ('_display_math_',                      'O'),
                ('for',                      'O'),
                ('every',                      'O'),]
        ed1, ed2 = zip(*expected_data2)

        words2, preds2 = llu.join_math_tokens(words, preds=preds)
        self.assertEqual(list(ed1), words2)
        self.assertEqual(list(ed2), preds2)

    def test_get_words_back_and_join_math_together3(self):
        in_data = [('[CLS]',                      'O'),
                ('A',                      'O'),
                ('mon',                      'O'),
                ('##oid',                      'O'),
                ('##al',                      'O'),
                ('category',                      'O'),
                ('is',                      'O'),
                ('symmetric',                      'B-DFNDUM'),
                ('mon',                      'I-DFNDUM'),
                ('##oid',                      'O'),
                ('##al',                      'O'),
                ('if',                      'O'),
                ('it',                      'O'),
                ('has',                      'O'),
                ('the',                      'O'),
                ('special',                      'O'),
                ('arrow',                      'O'),
                ('_',                      'O'),
                ('display',                      'O'),
                ('_',                      'O'),
                ('math',                      'O'),
                ('_',                      'O'),
                ('for',                      'O'),
                ('every',                      'O'),]
        expected_data = [ ('A',                      'O'),
                ('monoidal',                      'O'),
                ('category',                      'O'),
                ('is',                      'O'),
                ('symmetric',                      'B-DFNDUM'),
                ('monoidal',                      'I-DFNDUM'),
                ('if',                      'O'),
                ('it',                      'O'),
                ('has',                      'O'),
                ('the',                      'O'),
                ('special',                      'O'),
                ('arrow',                      'O'),
                ('_',                      'O'),
                ('display',                      'O'),
                ('_',                      'O'),
                ('math',                      'O'),
                ('_',                      'O'),
                ('for',                      'O'),
                ('every',                      'O'),]
        words, preds = llu.get_words_back(list(zip(*in_data))[0],
                list(zip(*in_data))[1],
                special_tokens=['[CLS]', '[PAD]'])
        ed1, ed2 = zip(*expected_data)
        self.assertEqual(list(ed1), words)
        self.assertEqual(list(ed2), preds)


        expected_data2 = [ ('A',                      'O'),
                ('monoidal',                      'O'),
                ('category',                      'O'),
                ('is',                      'O'),
                ('symmetric',                      'B-DFNDUM'),
                ('monoidal',                      'I-DFNDUM'),
                ('if',                      'O'),
                ('it',                      'O'),
                ('has',                      'O'),
                ('the',                      'O'),
                ('special',                      'O'),
                ('arrow',                      'O'),
                ('_display_math_',                      'O'),
                ('for',                      'O'),
                ('every',                      'O'),]
        ed1, ed2 = zip(*expected_data2)

        words2, preds2 = llu.join_math_tokens(words, preds=preds)
        self.assertEqual(list(ed1), words2)
        self.assertEqual(list(ed2), preds2)

    def test_get_entity(self):
        test_lst = [('a', 'B-DFNDUM'),
					('b', 'O'),
					('b', 'B-DFNDUM'),
					('b', 'O'),
					('b', 'O'),
					('b', 'O'),
					('b', 'O'),
					('b', 'B-DFNDUM'),
					('b', 'I-DFNDUM'),
					('b', 'I-DFNDUM'),]
        w, p = zip(*test_lst)
        expect_lst = llu.get_entity(w, p)
        self.assertEqual(expect_lst, ['a', 'b', 'b b b'])

