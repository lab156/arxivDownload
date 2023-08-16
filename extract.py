from lxml import etree
import parsing_xml
import pickle
from nltk.chunk import ChunkParserI
from ner.chunker import NamedEntityChunker, features
from nltk import pos_tag, word_tokenize
import nltk
#import sqlalchemy as sa
#from sqlalchemy.orm import sessionmaker
#from sampling import create_dict
import re
import numpy as np

from datasets import Dataset
from transformers import DataCollatorWithPadding

#TODO: All the xml tasks should be done by a helper
#function that just gets fed all the critical info
# filepath, number of articles, definitions
# If Definiendum gets no vectorizer, then it should not classify 

class Definiendum():
    def __init__(self, px, clf, bio, vzer, tzer, **kwargs):
        '''
        Extracts the definitions with of an xml/html file
        Arguments:
        `px`: parsing_xml.DefinitionsXML object
        `clf`: definition classifier  (all these objects should be unpickled once)
               if the prediction method returns probabilities need to include the 
               `thresh` parameter
        `bio`: iob or bio classifier
        `vzer`: word vectorizer
        `tzer`: tokenizer function ex.
                  def tok_function(example):
                      # This function can be used with the Dataset.map() method
                      return tokenizer(example['text'], truncation=True)

        Output: attrib .root is a XML tree with the format:
        <root>
          <article>
            <definition>
              <stmnt>
              <dfndum>
        '''
        min_words = kwargs.get('min_words', 15)
        self.px = px
        self.para_lst_idx = [(idx,p) for idx,p in 
                enumerate(map(px.recutext, px.para_list()))\
                         if len(p.split()) >= min_words]
        
        if self.para_lst_idx != []:
            # separate list of indices and text
            self.para_index, self.para_lst = list(zip(*(self.para_lst_idx)))
        else:
            self.para_lst = []
        self.vzer = vzer
        self.tzer = tzer
        assert (self.vzer is None) ^ (self.tzer is None),\
                print('vzer and tzer cannot be at the same time.')
        #self.clf = clf
        if bio is not None:
            self.chunk = lambda x: bio.parse(pos_tag(word_tokenize(self.clean_rm(x))))
        else:
            self.chunk = lambda x: []
        #first we need to vectorize

        thresh = kwargs.get('thresh', None)
        assert (tzer is None) or (thresh is None),\
            print('Incompatible parameters tzer not None means thresh should be None')

        if self.para_lst != []:
            if self.vzer is not None:
                self.trans_vect = vzer.transform(self.para_lst)
            elif self.tzer is not None:
                # create ds
                ds = Dataset.from_dict({
                    'text': self.para_lst,
                    'idx':  self.para_index,
                    })
                def tok_function(x):
                    # This function can be used with the Dataset.map() method
                    # x['text'] should be a paragraph
                    return tzer(x['text'], truncation=True)

                self.trans_vect = ds.map(tok_function, batched=True)
                data_coll = DataCollatorWithPadding(tokenizer=tzer, 
                        return_tensors='tf')
                self.trans_vect = self.trans_vect.to_tf_dataset(
                       columns=['attention_mask', 'input_ids', 'token_type_ids'],
                         shuffle=False,
                            collate_fn=data_coll,
                               batch_size=8 )
        else:
            # the nltk vectorizer raises this error on the on the empty list
            # OTOH, the NN vectorizer is normally implemented ad-hoc
            # This exception evens the behaviour
            raise ValueError('trying to vectorize empty para_lst.')


        if thresh is None:
            # This should be the case for HF LLM models
            self.predictions = clf.predict(self.trans_vect)
            if tzer is not None:
                # This extra step is needed for logit predictions
                self.predictions = np.argmax(self.predictions['logits'], axis=1)
        else:
            # clf.predict will give probabilities and thresh is the cutoff
            try:
                preds = clf.predict(self.trans_vect, batch_size=1)
                self.predictions = (preds > thresh).astype(int)
            except UnboundLocalError as ee:
                print(ee)
                print('length of trans_vect is: ', len(self.trans_vect))
            
        # Create list of pairs of definitions paired with the index 
        # in which they appear in the article
        self.def_lst = [p for ind, p in enumerate(self.para_lst_idx)
                if self.predictions[ind]]

        self.root = etree.Element('article')
        self.root.attrib['name'] = kwargs.get('fname', "")
        self.root.attrib['num'] = repr(len(px.para_list()))
        for ind,p in self.def_lst:
            defxml = self.create_definition_branch(ind, p)
            self.root.append(defxml)

    def get_definiendum(self, para):
        '''
        `para` is a nltk.tree.Tree For the index `k` return a list of definienda
        that is, the term being defined
        '''
        chunked = self.chunk(para)
        dfndum_lst = list(filter(lambda x: isinstance(x, nltk.tree.Tree), chunked))
        join_tokens = lambda D: ' '.join([d[0] for d in D])
        return [join_tokens(s) for s in dfndum_lst]

    def clean_rm(self, para):
        """
        Given a paragraph do the final cleanup before chunking
        This is basically removing the strings that are not in the
        wikipedia dataset due to their different origins:
        (wikipedia data is html and it's not produced by LaTeXML)
        """
        return re.sub(r"<s/>|</s>|_cite_|_citation_|_item_", "", para)

    def create_definition_branch(self, ind, defi):
        root = etree.Element("definition")
        root.attrib['index'] = repr(ind)
        statement = etree.SubElement(root, 'stmnt')
        statement.text = defi
        for d in self.get_definiendum(defi):
            dfndum = etree.SubElement(root, 'dfndum')
            dfndum.text = d
        return root


def query():
    eng = sa.create_engine('sqlite:///../arxiv1.db')
    eng.connect()
    SMaker = sa.orm.sessionmaker(bind=eng)
    sess = SMaker()
    return sess.execute('''SELECT id FROM articles
           where tags LIKE  '[{''term'': ''math.DG''%' and
           updated_parsed BETWEEN date('2015-01-01')  and date('2015-12-31');''')
        

if __name__ == '__main__':
    import sys
    import os
    import argparse
    parser = argparse.ArgumentParser(description='parsing xml commandline script')
    parser.add_argument('file_names', type=str, nargs='+',
            help='filenames to find definitions last position is the resulting files')
    parser.add_argument('-c', '--classifier',
            help='Path to the classifier pickle', type=str)
    parser.add_argument('-b', '--bio',
            help='Path to the BIO classfier pickle', type=str)
    parser.add_argument('-v', '--vectorizer',
            help='Path to the count vectorizer classfier pickle', type=str)
    parser.add_argument('-t', '--tokenizer',
            help='Path to the word tokenizer classfier pickle', type=str)
    parser.add_argument('-o', '--output',
            help='The output xml file to store everything', type=str)
    parser.add_argument('--query', action='store_true', 
            help='Ignore file_names and query')
    args = parser.parse_args(sys.argv[1:])

    with open(args.classifier, 'rb') as class_f:
        clf = pickle.load(class_f)
    with open(args.bio, 'rb') as class_f:
        bio = pickle.load(class_f)
    with open(args.vectorizer, 'rb') as class_f:
        vzer = pickle.load(class_f)
    with open(args.tokenizer, 'rb') as class_f:
        tokr = pickle.load(class_f)

    if args.output:
        '''
        Usage:
            python extract.py ~/media_home/math.AG/2015/*/*.xml -c ../PickleJar/classifier.pickle -b ../PickleJar/chunker.pickle -v ../PickleJar/vectorizer.pickle -t ../PickleJar/tokenizer.pickle -o ../mathAG_2015.xml
            '''

        try:
            root = etree.parse(args.output).getroot()
        except (OSError, etree.XMLSyntaxError):
            print(' File %s does not exist, will create later.'%args.output)
            root = etree.Element('root')
    else:
        # in this case the file does not exist yet
        root = etree.Element('root')

    if args.query:
        art_dict = create_dict()
        qq = query()
        change_path = lambda p: re.sub(r'^/mnt/', '/home/luis/media_home/', p)
        file_lst = [change_path(art_dict[s[0]]) for s in qq if s[0] in art_dict]
    else:
        file_lst = args.file_names

    for k,xml_path in enumerate(file_lst):
        havent_done = root.find('.//article[@name = "%s"]'%xml_path) is None
        if havent_done:
            print('Processing file: %s'%os.path.basename(xml_path), end='\r')
            try:
                px = parsing_xml.DefinitionsXML(xml_path)
                ddum = Definiendum(px, clf, bio, vzer, tokr)
                root.append(ddum.root)
                if k%25 == 0 and args.output:
                    with open(args.output, 'w') as out_f:
                        out_f.write(etree.tostring(root, pretty_print=True)\
                                .decode('utf8'))
            except (TypeError, etree.ParseError):
                print('file %s could not be parsed by parsing_xml'%os.path.basename(
                    xml_path))
            except ValueError as e:
                print('In the file %s found the problem'%os.path.basename(xml_path))
                print(e)
        else:
            print('Already did file: %s'%os.path.basename(xml_path), end='\r')
    else:
        if args.output:
            with open(args.output, 'w') as out_f:
                print(etree.tostring(root, pretty_print=True).decode('utf8'), file=out_f)
        else:
        # If no output was specified then print the result
        # This might get very big so not sure if this is the right way
            print(etree.tostring(root, pretty_print=True).decode('utf8') )


