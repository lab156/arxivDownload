from nltk.chunk import ChunkParserI
from nltk.stem.snowball import SnowballStemmer
import string
from nltk.chunk import conlltags2tree, tree2conlltags
from collections.abc import Iterable
from nltk.tag import ClassifierBasedTagger

class NamedEntityChunker(ChunkParserI):
    def __init__(self, train_sents, **kwargs):
        assert isinstance(train_sents, Iterable)
 
        self.feature_detector = features
        self.tagger = ClassifierBasedTagger(
            train=train_sents,
            feature_detector=features,
            **kwargs)
    def parse(self, tagged_sent):
        chunks = self.tagger.tag(tagged_sent)
 
        # Transform the result from [((w1, t1), iob1), ...] 
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]
 
        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets)       


def features(tokens, index, history):
    """
    `tokens`  = a POS-tagged sentence [(w1, t1), ...]
    `index`   = the index of the token we want to extract features for
    `history` = the previous predicted IOB tags
    """
 
    # init the stemmer
    stemmer = SnowballStemmer('english')
 
    # Pad the sequence with placeholders
    tokens = [('[START3]', '[START3]'),('[START2]', '[START2]'), ('[START1]', '[START1]')] +\
    list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]'), ('[END3]', '[END3]')]
    history = ['[START3]', '[START2]', '[START1]'] + list(history)
 
    # shift the index with 3, to accommodate the padding
    index += 3
 
    word, pos = tokens[index]
    prevword, prevpos = tokens[index - 1]
    prevprevword, prevprevpos = tokens[index - 2]
    prev3word, prev3pos = tokens[index - 3]
    nextword, nextpos = tokens[index + 1]
    nextnextword, nextnextpos = tokens[index + 2]
    next3word, next3pos = tokens[index + 3]
    previob = history[index - 1]
    prevpreviob = history[index - 2]
    prev3iob = history[index - 3]
    contains_dash = '-' in word
    contains_dot = '.' in word
    allascii = all([True for c in word if c in string.ascii_lowercase])
 
    allcaps = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase
 
    prevallcaps = prevword == prevword.capitalize()
    prevcapitalized = prevword[0] in string.ascii_uppercase
 
    nextallcaps = prevword == prevword.capitalize()
    nextcapitalized = prevword[0] in string.ascii_uppercase
    
    is_math = lambda w:(w == '_inline_math_') or (w == '_display_math_')
    ismath = is_math(word)
    isprevmath = is_math(prevword)
    isprevprevmath = is_math(prevprevword)
 
    return {
        'word': word,
        'lemma': stemmer.stem(word),
                'pos': pos,
        'all-ascii': allascii,
 
        'next-word': nextword,
        'next-lemma': stemmer.stem(nextword),
        'next-pos': nextpos,
 
        'next-next-word': nextnextword,
        'nextnextpos': nextnextpos,
 
        'next3word': next3word,
        'next3pos': next3pos,
        
        'prev-word': prevword,
        'prev-lemma': stemmer.stem(prevword),
        'prev-pos': prevpos,
 
        'prev-prev-word': prevprevword,
        'prev-prev-pos': prevprevpos,
 
        'prev3word': prev3word,
        'prev3pos': prev3pos,
        
        'prev-iob': previob,
        
        'prev-prev-iob': prevpreviob,
 
        'contains-dash': contains_dash,
        'contains-dot': contains_dot,
 
        'all-caps': allcaps,
        'capitalized': capitalized,
 
        'prev-all-caps': prevallcaps,
        'prev-capitalized': prevcapitalized,
 
        'next-all-caps': nextallcaps,
        'next-capitalized': nextcapitalized,
        
        'ismath': ismath,
        'isprevmath': isprevmath,
        'isprevprevmath': isprevprevmath,
    }

