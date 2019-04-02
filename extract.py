from lxml import etree
import parsing_xml 
import pickle
from nltk.chunk import ChunkParserI
from ner.chunker import NamedEntityChunker, features
from nltk import pos_tag, word_tokenize
import nltk


class Definiendum():
    def __init__(self, para_lst, clf, bio, vzer, tzer):
        '''
        Extracts the definitions with of an xml/html file
        `para_lst`: list of clean paragraphs (ready to vecorize)
        `clf`: definition classifier  (all these objects should be unpickled once)
        `bio`: iob or bio classifier
        `vzer`: word vectorizer

        Output is a dictionary of a list of definiendum, statement of the definition
        paragraph number
        '''
        self.para_lst = para_lst
        self.vzer = vzer
        #self.clf = clf
        self.chunk = lambda x: bio.parse(pos_tag(word_tokenize(x)))
        #first we need to vectorize 
        
        self.trans_vect = vzer.transform(para_lst)
        self.predictions = zip(clf.predict(self.trans_vect), para_lst)
        #print(len(self.predictions))
        self.def_lst = [p for pred, p in zip(self.predictions, para_lst) if pred]
        print(self.chunk(self.def_lst[0]))
        print(self.get_definiendum(0))
        for k,p in enumerate(self.predictions):
            if p:
                pass

    def get_definiendum(self, k):
        chunked = self.chunk(self.def_lst[k])
        dfndum_lst = list(filter(lambda x: isinstance(x, nltk.tree.Tree), chunked))
        flatten = lambda D: ' '.join([d[0] for d in D])
        return [flatten(s) for s in dfndum_lst]
        
        


if __name__ == '__main__':
    import sys

    with open('data/PickleJar/class.pickle', 'rb') as class_f:
        clf = pickle.load(class_f)
    with open('data/PickleJar/chunker.pickle', 'rb') as class_f:
        bio = pickle.load(class_f)
    with open('data/PickleJar/vectorizer.pickle', 'rb') as class_f:
        vzer = pickle.load(class_f)
    with open('data/PickleJar/tokenizer.pickle', 'rb') as class_f:
        tokr = pickle.load(class_f)

    px = parsing_xml.DefinitionsXML('data/examples/1501.06563/1501.06563.xml')
    para_lst = list(map(px.recutext, px.para_list()))
    ddum = Definiendum(para_lst, clf, bio, vzer, tokr)

