import parsing_xml as px
import xml.etree.ElementTree as ET
import random
import sys
import magic
import tarfile
import logging

'''
This file contains routines to sample from 
the same articles that we mine for NEGATIVE examples 
to train on
'''

ns = {'latexml': 'http://dlmf.nist.gov/LaTeXML' }


def sample_article(f, ns, para_per_article=2, min_words=15):
    '''
    Usage: f be a parsable xml tree
    try to get para_per_article paragraphs from this article
    min_words: the paragraph has to have more that this amount of words
    '''
    try:
        exml = ET.parse(f)
        para_lst_nonrand = exml.findall('.//latexml:para',ns)
        para_lst = random.sample(para_lst_nonrand,
                para_per_article)
    except ET.ParseError:
        print('article %s could no be parsed'%f)
        para_lst = []
    except ValueError:
        print('article %s has few paragraphs'%f)
        para_lst = []

    return_lst = []
    for p in para_lst:
        if px.check_sanity(p, ns):
            para_text =  px.recutext_xml(p)
            if len(para_text.split()) >= min_words: #check min_words
                return_lst.append(para_text)
        else:
            print('article %s has messed up para'%f)
    return return_lst

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='parsing xml commandline script')

    parser.add_argument('file_names', type=str, nargs='+',
            help='filenames to find definitions last position is the resulting files')

    parser.add_argument('-o', '--outfile', help='file to write the random paragraphs',
            type=str)

    args = parser.parse_args(sys.argv[1:])

    file_list = args.file_names

    if args.outfile:
        # open the file only once for append
        with open(args.outfile, 'a') as oa:
            for k,f in enumerate(file_list):
                p_lst = sample_article(f, ns)
                print("Writing %2d parag in file: %s number %3d/%d                    "\
                        %(len(p_lst),
                    '/'.join(f.split('/')[-2:]),
                    k, len(file_list)), end='\r', flush=True)
                for p in p_lst:
                    oa.write(p + '\n')
        print("------------")
    else:
        for f in file_list:
            print(":::::::::::: %s ::::::::::::::::"%f)
            if magic.detect_from_filename(f).mime_type == 'application/gzip':
                with tarfile.open(tarpath) as tar_file:
                    for pathname in tar_file.getnames():
                        dirname = pathname.split('/')[1]
                        article_dict[dirname].append(pathname)
                    for name,val in article_dict.items():
                        comm = tar_file.extractfile(next(filter('.tar.gz', val)))
                    p_lst = sample_article(f, ns)
                print('------------------------------')

