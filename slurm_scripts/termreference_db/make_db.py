from lxml import etree
import pickle
import numpy as np
from collections import Counter, defaultdict
import glob
from tqdm import tqdm
import tarfile
from tqdm import tqdm
from marshmallow import Schema, fields, pprint
from random import random
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import datetime as dt
import toml
config = toml.load('../../config.toml')

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, os.path.join(parentdir, 'embed'))
sys.path.insert(0, parentdir)

from clean_and_token_text import normalize_text, normalize_phrase, join_xml_para_and_write, ReadGlossary
import parsing_xml as px
import peep_tar as peep
from enum import Enum

def gen_termreferences(glossary_file_lst, data_path):
    #glossary_file_lst = glob.glob('/media/hd1/glossary/NN.v1/math9*/*.xml.gz')
    #glossary_file_lst = glob.glob('/media/hd1/glossary/NN.v1/math0*/*.xml.gz')

    RG = ReadGlossary(os.path.join(data_path, 'glossary/v3/math*/*.xml.gz'),
            os.path.join(data_path, 'glossary/NN.v1/math*/*.xml.gz'))
    vocab = RG.ntc_intersect('relative')

    corpus = []
    term_ref_lst = []
    for glossary_file in tqdm(glossary_file_lst):
        # argot_file format: /media/hd1/glossary/NN.v1/math95/9506_001.xml.gz
        math_year = glossary_file.split('/')[-2]
        subfilename = glossary_file.split('/')[-1].split('.')[0] # should be 9506_001
        promath_file = os.path.join(data_path, 'promath', math_year, subfilename + '.tar.gz')
        joined_file = os.path.join(data_path, 'cleaned_text/joined_math19-35_13-01/',
                                   math_year, subfilename + '.xml.gz')
        
        glossary_xml = etree.parse(glossary_file)
        joined_xml = etree.parse(joined_file)
        promath_tarfobj = tarfile.open(promath_file)
        for art in glossary_xml.findall('./article'):
            art_name = art.get('name')
            # need to use only the basename of the article
            #art_basename = os.path.basename(art_name)
            joined_name_results = joined_xml.xpath('./article[@name="{}"]'.format(art_name))
            # POPULATE THE CORPUS
            if len(joined_name_results) > 0:
                joined_art_text = ''
                for parag in joined_name_results[0].findall('./parag'):
                    text = '' if parag.text is None else parag.text
                    joined_art_text += (text + ' ')
                corpus.append(joined_art_text)
            else:
                print(f'joined name search results empty art_name={art_name}')
                corpus.append('')
            corpus_index = len(corpus) - 1
            
            # OPEN THE PROMATH FILE
            try:
                #promath_obj = peep.tar(promath_file, art_name)[1].exml
                promath_obj = px.DefinitionsXML(promath_tarfobj.extractfile(art_name))
            except AttributeError:
                print(f'{art_name} gave attributeError')
            # this list is in sync with glossary paragraph index
            promath_parag_lst = promath_obj.para_list()
            
            # LOOP THROUGH THE DFDUMS
            for defin in art.findall('./definition'):
                p_index = int(defin.get('index'))
                for term_raw in defin.findall('./dfndum'):
                    term = normalize_text(term_raw.text)
                    if term in vocab:
                        term_ref_lst.append((
                               term,
                                art_name,
                                p_index,
                                etree.tostring(promath_parag_lst[p_index]).decode('utf-8'),
                                corpus_index))
        promath_tarfobj.close()
    return corpus, term_ref_lst, vocab
        
    

def gen_tfidf(vocab, corpus):
    # COMPUTE THE TDIDF MATRIX
    vocab_ = list(set([t.replace(' ', '_') for t in vocab]))
    tvect = TfidfVectorizer(sublinear_tf=True, norm='l1', vocabulary=vocab_)
                    
    Now = dt.datetime.now()
    ttrans = tvect.fit_transform(corpus)
    tdelta = dt.datetime.now() - Now
    print(f'The shape of the resulting matrix is: {ttrans.shape} and it took {tdelta.seconds} seconds')
    return ttrans, vocab_

def get_tfidf(term_, corpus_index_, vocab_, ttrans):
    ter = term_.replace(' ', '_')
    tindex = vocab_.index(ter)
    x = ttrans[corpus_index_, tindex]
    return int(x * 10_000_000_000)
# get_tfidf('infinite _inline_math_cluster', 3080, vocab_)

@dataclass
class TermReference(Schema):
    truid: int
    term : str
    addr : str
    index : int
    p_tag : str
    tfidf : int
    
class TermRefSchema(Schema):
    truid = fields.Int()
    term =  fields.String()
    addr =  fields.String()
    index = fields.Int()
    p_tag = fields.String()
    tfidf = fields.Int()

def write_data(term_ref_lst, vocab, out_dir):
    print('saving term_ref_lst ')
    with open(os.path.join(out_dir, 'term_ref_lst.pickle'), 'wb') as fobj:
        pickle.dump(term_ref_lst, fobj)
    print('saving vocab ')
    with open(os.path.join(out_dir, 'vocab.pickle'), 'wb') as fobj:
        pickle.dump(vocab, fobj)
    print('saving corpus ')
    with open(os.path.join(out_dir, 'corpus.pickle'), 'wb') as fobj:
        pickle.dump(corpus, fobj)
    
def write_json(term_ref_lst, vocab_, out_file_path, ttrans):
    TR_lst=[]
    for k, tr in enumerate(term_ref_lst):
        # check the length of p_tag
        if len(tr[3]) < 250_000:
            TR_lst.append(TermReference(
                truid = k,
                term= tr[0],
                   addr = tr[1],
                   index = tr[2],
                  p_tag = tr[3],
                   tfidf = get_tfidf(tr[0], tr[4], vocab_, ttrans) + tr[2]))
    # adding tfidf + index has three nice properties
    #1) together with term it is mostly unique 
    #2) if a term appears twice on the same paragraph, it gets rewritten and merges in the data
    #3) it sorts in the order in which the term appears on the same article.
    # Serialize all the data and save to json file
    trs = TermRefSchema(many=True)
    tr_ = TermRefSchema()
    with open(out_file_path, 'w') as fobj:
        fobj.write(trs.dumps(TR_lst))

def main():
    '''
    example:
    python3 make_db.py "/media/hd1/glossary/NN.v1/math9*/*.xml.gz" /tmp/rm_me_hola
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('in_files', type=str, 
            help='glob expression from the files in glossary ex. $PROJ"/glossary/NN.v1/math9*/*.xml.gz"')
    parser.add_argument('out_dir', type=str,
            help='Directory to write out the json file')
    parser.add_argument('--data_path', default=None,
            help='Path to the data files in case there is something faster ex. $LOCAL/')
    args = parser.parse_args()

    if args.data_path is not None:
        data_path = args.data_path
    else:
        data_path = config['paths']['data']

    glossary_file_lst = glob.glob(args.in_files)
    print(f"{len(glossary_file_lst)} glossary files have been selected for processing")

    corpus, term_ref_lst, vocab = gen_termreferences(glossary_file_lst, data_path)

    #ttrans, vocab_ = gen_tfidf(vocab, corpus)

    os.makedirs(args.out_dir, exist_ok=True)
    out_file_path = os.path.join(args.out_dir, 'termrefs.json')
    write_data(term_ref_lst, vocab, corpus, args.out_dir)
    #write_json(term_ref_lst, vocab_, out_file_path, ttrans)


if __name__ == "__main__":
    main()

