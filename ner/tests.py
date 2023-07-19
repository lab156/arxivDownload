import unittest
try:
    import bio_tag as iob
    import llm_utils as lu
except ModuleNotFoundError:
    from ner import bio_tag as iob
    from ner import llm_utils as lu

#RUN WITH: python -m unittest tests.py 

class dummy_tok:
    def tokenize(self, s):
        return [s]

class TestLlmUtils(unittest.TestCase):
    def test_get_words_back(self):
        special_toks = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
        test1 = ['Hi', 'my', 'Na', '##me']
        gold1 = ['Hi', 'my', 'Name']
        t1, p1 = lu.get_words_back(test1)
        self.assertEqual(gold1, t1)

        in_words, in_preds = (['[CLS]', 'More', 'precisely', ',', 'a', 'unitary', 'transformation', 'is', 'an', 'is', '##omo', '##rp', '##hism', 'between', 'two', 'Hi', '##lbert', 'spaces', '.', '[SEP]'],   
				['O', 'O', 'O', 'O', 'O', 'B-defndum', 'I-defndum', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])
                 
        out_words, out_preds = (['More', 'precisely', ',', 'a', 'unitary', 'transformation', 'is', 'an', 'isomorphism', 'between', 'two', 'Hilbert', 'spaces', '.'],
 ['O', 'O', 'O', 'O', 'B-defndum', 'I-defndum', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'])
        o_w, o_p = lu.get_words_back(in_words, in_preds, special_tokens=special_toks)
        self.assertEqual(out_words, o_w)
        self.assertEqual(out_preds, o_p)

    def test_get_words_back2(self):
        special_toks = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']

        in_words, in_preds = (['[CLS]', 'a', 'b', '##b', '##b'],
                              ['O',     'O', 'B-def', 'O', 'I-def'])
                 
        out_words, out_preds = (['a', 'bbb'],
                                ['O', 'B-def'])
 
        o_w, o_p = lu.get_words_back(in_words, in_preds, special_tokens=special_toks)
        self.assertEqual(out_words, o_w)
        self.assertEqual(out_preds, o_p)

    def test_get_words_back3(self):
        special_toks = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']

        in_words, in_preds = (['a', '##b', '##b', '##b'],
                              ['O', 'B-def', 'O', 'I-def'])
                 
        out_words, out_preds = (['abbb'],
                                ['O',])
 
        o_w, o_p = lu.get_words_back(in_words, in_preds, special_tokens=special_toks)
        self.assertEqual(out_words, o_w)
        self.assertEqual(out_preds, o_p)
    
    def test_join_math_tokens_inline(self):
        test_inline = ['_',
			 'inline',
			 '_',
			 'math',
			 '_', ]
        out, pred = lu.join_math_tokens(test_inline)
        gold = ['_inline_math_']
        self.assertEqual(gold, out)
        self.assertEqual(['O'], pred)
        
    def test_join_math_tokens_display(self):
        test_inline = ['_',
			 'display',
			 '_',
			 'math',
			 '_', ]
        out, pred = lu.join_math_tokens(test_inline)
        gold = ['_display_math_']
        self.assertEqual(gold, out)
        self.assertEqual(['O'], pred)
        
    def test_join_math_tokens_double(self):
        test_inline = ['_',
			 'display',
			 '_',
			 'math',
			 '_', 
             '_',
			 'display',
			 '_',
			 'math',
			 '_', ]
        out, pred = lu.join_math_tokens(test_inline)
        gold = ['_display_math_', '_display_math_']
        self.assertEqual(gold, out)
        self.assertEqual(['O', 'O'], pred)

    def test_join_math_tokens_incomplete(self):
        test_inline = ['_',
			 'display',
			 '_',
			 'math',
			 '_', 
             '_',
			 'display',
			 '_',
			 '_', ]
        out, pred = lu.join_math_tokens(test_inline)
        gold = ['_display_math_', '_', 'display', '_', '_']
        self.assertEqual(gold, out)
        self.assertEqual(['O', 'O', 'O', 'O', 'O'], pred)

    def test_join_math_tokens_incomplete2(self):
        test_inline = ['_',
			 'display',
			 '_',
			 'math',
			 '_', 
             '_',
			 'display',
			 '_',
             'math',
			  ]
        out, pred = lu.join_math_tokens(test_inline)
        gold = ['_display_math_', '_', 'display', '_', 'math']
        self.assertEqual(gold, out)
        self.assertEqual(['O', 'O', 'O', 'O', 'O'], pred)

class TestBIOTagger(unittest.TestCase):
    def test_short_sentence(self):
        sent = [('hello', 'NN')]
        title = 'extremely long title'
        p1 = [('hello', 'NN', 'O')]
        p2 = iob.bio_tagger(title.split(), sent)
        self.assertEqual(p1, p2)

    def test_super_easy(self):
        sent = [('A', 'DT'), ('triangle', 'NN'), ('has', 'VBZ'),
                ('three', 'CD'), ('sides', 'NNS'), ('.', '.')]
        title = 'triangle'
        p1 =[('A', 'DT', 'O'),
             ('triangle', 'NN', 'B-DFNDUM'),
             ('has', 'VBZ', 'O'),
             ('three', 'CD', 'O'),
             ('sides', 'NNS', 'O'),
             ('.', '.', 'O')]
        p2 = iob.bio_tagger(title.split(), sent)
        self.assertEqual(p1, p2)

    def test_at_the_beginning(self):
        sent = [('right', 'RB'), ('triangle', 'NN'), ('appears', 'VBZ'),
                ('at', 'IN'), ('the', 'DT'), ('beginning', 'NN')]
        title = 'right triangle'
        p1 = [('right', 'RB', 'B-DFNDUM'),
             ('triangle', 'NN', 'I-DFNDUM'),
             ('appears', 'VBZ', 'O'),
             ('at', 'IN', 'O'),
             ('the', 'DT', 'O'),
             ('beginning', 'NN', 'O')]
        p2 = iob.bio_tagger(title.split(), sent)
        self.assertEqual(p1, p2)

    def test_appears_twice(self):
        sent = [('a', 'DT'), ('prime', 'JJ'), ('number', 'NN'),
                ('is', 'VBZ'), ('only', 'RB'), ('divisible', 'JJ'),
                ('by', 'IN'), ('1', 'CD'), ('and', 'CC'), ('a', 'DT'),
                ('prime', 'JJ'), ('number', 'NN')]
        title = 'prime number'
        p1 = [('a', 'DT', 'O'),
             ('prime', 'JJ', 'B-DFNDUM'),
             ('number', 'NN', 'I-DFNDUM'),
             ('is', 'VBZ', 'O'),
             ('only', 'RB', 'O'),
             ('divisible', 'JJ', 'O'),
             ('by', 'IN', 'O'),
             ('1', 'CD', 'O'),
             ('and', 'CC', 'O'),
             ('a', 'DT', 'O'),
             ('prime', 'JJ', 'B-DFNDUM'),
             ('number', 'NN', 'I-DFNDUM')]
        p2 = iob.bio_tagger(title.split(), sent)
        self.assertEqual(p1, p2)

    def test_crazy_example(self):
        sent = [('the', 'DT'), ('technology', 'NN'), ('wiki', 'NN'),
                ('wiki', 'NN'), ('wiki', 'NN'), ('is', 'VBZ'),
                ('very', 'RB'), ('interesting', 'JJ'), ('wiki', 'NN'),
                ('wiki', 'NN')]
        title = 'wiki wiki'
        p1 = [('the', 'DT', 'O'),
             ('technology', 'NN', 'O'),
             ('wiki', 'NN', 'B-DFNDUM'),
             ('wiki', 'NN', 'I-DFNDUM'),
             ('wiki', 'NN', 'O'),
             ('is', 'VBZ', 'O'),
             ('very', 'RB', 'O'),
             ('interesting', 'JJ', 'O'),
             ('wiki', 'NN', 'B-DFNDUM'),
             ('wiki', 'NN', 'I-DFNDUM')]
        p2 = iob.bio_tagger(title.split(), sent)
        self.assertEqual(p1, p2)

    def test_common_plural_case(self):
        sent = [('The', 'DT'), 
                  ('set', 'NN'),
                  ('of', 'IN'),
                  ('hybrid', 'JJ'),
                  ('numbers', 'NNS'),
                  ('_inline_math_', 'VBP'),]
        title = 'Hybrid number'
        p1 = [('The', 'DT', 'O'),
              ('set', 'NN', 'O'),
              ('of', 'IN', 'O'),
              ('hybrid', 'JJ', 'B-DFNDUM'),
              ('numbers', 'NNS', 'I-DFNDUM'),
              ('_inline_math_', 'VBP', 'O'),]
        p2 = iob.bio_tagger(title.split(), sent)
        self.assertEqual(p1, p2)

    def test_put_pos_ner_tags(self):
        in_data = ('Government hacking', '', "One such option is the so-called government hacking.")
        out_data = [{'title': 'Government hacking',
                      'section': '',
                      'defin': 'One such option is the so-called government hacking.',
                      'ner': [(('One', 'CD'), 'O'),
                       (('such', 'JJ'), 'O'),
                       (('option', 'NN'), 'O'),
                       (('is', 'VBZ'), 'O'),
                       (('the', 'DT'), 'O'),
                       (('so-called', 'JJ'), 'O'),
                       (('government', 'NN'), 'B-DFNDUM'),
                       (('hacking', 'NN'), 'I-DFNDUM'),
                       (('.', '.'), 'O')]}]
        check = iob.put_pos_ner_tags([in_data], dummy_tok())
        self.assertEqual(check, out_data)

    def test_bio_tkn_tagger_basic(self):
        title = 'hola macizo'.split()
        sentence = 'como estal hola macizo mera verga o hola hola sos el hola macizo'.split()
        out_data = [('como', 0),
                 ('estal', 0),
                  ('hola', 1),
                   ('macizo', 2),
                    ('mera', 0),
                     ('verga', 0),
                      ('o', 0),
                       ('hola', 0),
                        ('hola', 0),
                         ('sos', 0),
                          ('el', 0),
                           ('hola', 1),
                            ('macizo', 2)]
        check = iob.bio_tkn_tagger(title, sentence)
        self.assertEqual(check, out_data)

    def test_bio_tkn_tagger_cap_unsensitive1(self):
        title = 'hola Macizo'.split()
        sentence = 'como estal hola macizo mera verga o hola hola sos el hola macizo'.split()
        out_data = [('como', 0),
                 ('estal', 0),
                  ('hola', 1),
                   ('macizo', 2),
                    ('mera', 0),
                     ('verga', 0),
                      ('o', 0),
                       ('hola', 0),
                        ('hola', 0),
                         ('sos', 0),
                          ('el', 0),
                           ('hola', 1),
                            ('macizo', 2)]
        check = iob.bio_tkn_tagger(title, sentence)
        self.assertEqual(check, out_data)

    def test_bio_tkn_tagger_cap_unsensitive2(self):
        title = 'hola macizo'.split()
        sentence = 'como estal Hola macizo mera verga o hola hola sos el hola Macizo'.split()
        out_data = [('como', 0),
                 ('estal', 0),
                  ('Hola', 1),
                   ('macizo', 2),
                    ('mera', 0),
                     ('verga', 0),
                      ('o', 0),
                       ('hola', 0),
                        ('hola', 0),
                         ('sos', 0),
                          ('el', 0),
                           ('hola', 1),
                            ('Macizo', 2)]
        check = iob.bio_tkn_tagger(title, sentence)
        self.assertEqual(check, out_data)

    def test_join_by_example_equal_args(self):
        tokens = ['The', 'money', 'multiplier', 'is', 'defined', 'in', 'various', 'ways', '.']
        golds = ['The', 'money', 'multiplier', 'is', 'defined', 'in', 'various', 'ways', '.']
        out1, out2 = lu.join_by_example(tokens, golds)
        self.assertEqual(out1, golds)
        self.assertEqual(len(out2), len(golds))

    def test_join_by_example_equal_args2(self):
        tokens = ['The', 'money', 'multiplier', 'is', 'defined', 'in', 'various', 'ways', '.']
        preds = ['O', 'B-DFNDUM', 'I-DFNDUM', 'O', 'O', 'O', 'O', 'O', 'O']
        golds = ['The', 'money', 'multiplier', 'is', 'defined', 'in', 'various', 'ways', '.']
        out1, out2 = lu.join_by_example(tokens, golds, preds=preds)
        self.assertEqual(out1, golds)
        self.assertEqual(len(out2), len(golds))

    def test_join_by_example_spliter_and_join(self):
        tokens = ['The', 'money', 'mul', 'tip', 'lier', 'is', 'defined', 'in', 'various', 'ways', '.']
        preds = ['O', 'B-DFNDUM',  'I-DFNDUM', 'I-DFNDUM','I-DFNDUM', 'O', 'O', 'O', 'O', 'O', 'O']
        golds = ['The', 'money', 'multiplier', 'is', 'defined', 'in', 'various', 'ways', '.']
        gold_labels = ['O', 'B-DFNDUM', 'I-DFNDUM', 'O', 'O', 'O', 'O', 'O', 'O']
        out1, out2 = lu.join_by_example(tokens, golds, preds=preds)
        self.assertEqual(out1, golds)
        self.assertEqual(len(out2), len(golds))
        self.assertEqual(out2, gold_labels)

    def test_join_by_example_merge_labels(self):
        tokens = ['The', 'money', 'mul', 'tip', 'lier', 'is', 'defined', 'in', 'various', 'ways', '.']
        preds = ['O', 'O',  'I-DFNDUM', 'B-DFNDUM','O', 'O', 'O', 'O', 'O', 'O', 'O']
        golds = ['The', 'money', 'multiplier', 'is', 'defined', 'in', 'various', 'ways', '.']
        gold_labels = ['O', 'O', 'B-DFNDUM', 'O', 'O', 'O', 'O', 'O', 'O']
        out1, out2 = lu.join_by_example(tokens, golds, preds=preds)
        self.assertEqual(out1, golds)
        self.assertEqual(len(out2), len(golds))
        self.assertEqual(out2, gold_labels)

    def test_join_by_example_double_join(self):
        tokens = ['The', 'mon', 'ey', 'mul', 'tip', 'lier', 'is', 'defined', 'in', 'various', 'ways', '.']
        preds = ['O', 'O', 'B-DFNDUM',  'I-DFNDUM', 'I-DFNDUM','I-DFNDUM', 'O', 'O', 'O', 'O', 'O', 'O']
        golds = ['The', 'money', 'multiplier', 'is', 'defined', 'in', 'various', 'ways', '.']
        gold_labels = ['O', 'B-DFNDUM', 'I-DFNDUM', 'O', 'O', 'O', 'O', 'O', 'O']
        out1, out2 = lu.join_by_example(tokens, golds, preds=preds)
        self.assertEqual(out1, golds)
        self.assertEqual(len(out2), len(golds))
        self.assertEqual(out2, gold_labels)





if __name__ == '__main__':
    # RUN TESTS WITH python tests.py
    unittest.main()
