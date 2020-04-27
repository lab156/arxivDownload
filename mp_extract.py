import multiprocessing as mp
from extract import Definiendum
import parsing_xml
from lxml import etree
import pickle
from nltk.chunk import ChunkParserI
from ner.chunker import NamedEntityChunker, features
from nltk import pos_tag, word_tokenize
import tarfile
import gzip
import logging
import os

# Classify paragraphs and extract definitions and write results to .xml format
# In parallel using several processors

def parse_clf_chunk(file_obj, clf, bio, vzer, tokr):
    '''
    Runs the classifier and chunker on the file_obj
    file_obj: file object

    clf, bio, vzer, tokr: pickled classifiers and tokenizer
    '''
    px = parsing_xml.DefinitionsXML(file_obj)
    ddum = Definiendum(px, clf, bio, vzer, tokr)
    return ddum.root

def untar_clf_write(tarpath, output_dir,  *args):
    '''
    Takes a tarfile and runs parse_class_chunk and writes to output_dir

    tarpath: the path to the tar file with all the articles inside ex.
    1401_001.tar.gz

    output_dir: a path to write down results
    '''
    root = etree.Element('root')
    logging.info('Started working on tarpath=%s'%tarpath)
    with tarfile.open(tarpath) as tar_file:
        xml_file_lst = [t for t in tar_file.getnames() if t.endswith('.xml')]
        for xml_file in xml_file_lst: 
            try:
                tar_fileobj = tar_file.extractfile(xml_file)
                art_tree = parse_clf_chunk(tar_fileobj, *args)
                root.append(art_tree)
            except ValueError as ee:
                logging.debug(' '.join([repr(ee), 'file: ', xml_file, ' is empty']))

    #name of tarpath should be in the format '1401_001.tar.gz'
    gz_filename = os.path.basename(tarpath).split('.')[0] + '.xml.gz' 
    logging.debug('The name of the gz_filename is: %s'%gz_filename)
    gz_out_path = os.path.join(output_dir, gz_filename) 
    with gzip.open(gz_out_path, 'wb') as out_f:
        logging.info("Writing to dfdum zipped file to: %s"%gz_out_path)
        out_f.write(etree.tostring(root, pretty_print=True))



if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Given a Directory with the\
            structure from the arxiv, run the classifier and chunker in\
            parallel')
    parser.add_argument('dirnames', type=str, nargs='+',
            help='path to dir with the article files: ex. math05')
    parser.add_argument('-c', '--classifier',
            help='Path to the classifier pickle', type=str)
    parser.add_argument('-b', '--bio',
            help='Path to the BIO classfier pickle', type=str)
    parser.add_argument('-v', '--vectorizer',
            help='Path to the count vectorizer classfier pickle', type=str)
    parser.add_argument('-t', '--tokenizer',
            help='Path to the word tokenizer classfier pickle', type=str)
    parser.add_argument('-o', '--output',
            help='The output directory to store the definition definieda xml', type=str)
    parser.add_argument('--query', action='store_true', 
            help='Ignore file_names and query')
    parser.add_argument("--log", type=str, default=None,
            help='log level, options are: debug, info, warning, error, critical')
    args = parser.parse_args(sys.argv[1:])

    if args.log:
        logging.basicConfig(level=getattr(logging, args.log.upper()))

    with open(args.classifier, 'rb') as class_f:
        clf = pickle.load(class_f)
    with open(args.bio, 'rb') as class_f:
        bio = pickle.load(class_f)
    with open(args.vectorizer, 'rb') as class_f:
        vzer = pickle.load(class_f)
    with open(args.tokenizer, 'rb') as class_f:
        tokr = pickle.load(class_f)

    pool = mp.Pool(processes=4)
    f_lst = args.dirnames
    out_path = args.output
    arg_lst = [(t, out_path, clf, bio, vzer, tokr) for t in f_lst]
    pool.starmap(untar_clf_write, arg_lst)
