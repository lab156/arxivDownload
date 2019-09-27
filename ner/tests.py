import unittest
import bio_tag as iob

#RUN WITH: python -m unittest tests.py 

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


if __name__ == '__main__':
    # RUN TESTS WITH python tests.py
    unittest.main()
