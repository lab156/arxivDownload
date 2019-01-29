import parsing_xml as px
import xml.etree.ElementTree as ET
import random
import sys

'''
This file contains routines to sample from 
the same articles that we mine. 
In search of the negative examples to train on
'''

ns = {'latexml': 'http://dlmf.nist.gov/LaTeXML' }

def check_sanity(p):
    '''
    Input:
    p: element tree result of searching for para tags
    Checks:
    contains an ERROR tag
    '''
    if p.findall('.//latexml:ERROR', ns):
        return False
    else:
        return True

def sample_article(f, para_per_article=2, min_words=15):
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
        para_lst = []
    except ValueError:
        print('article %s has few paragraphs'%f)

    return_lst = []
    for p in para_lst:
        if check_sanity(p):
            para_text =  px.recutext1(p)
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
                p_lst = sample_article(f)
                print("Writing %2d parag in file: %s number %3d/%d               "%(len(p_lst),
                    '/'.join(f.split('/')[-2:]), k, len(file_list)), end='\r', flush=True)
                for p in p_lst:
                    oa.write(p + '\n')
        print("------------")
    else:
        for f in file_list:
            print(":::::::::::: %s ::::::::::::::::"%f)
            p_lst = sample_article(f)
            for p in p_lst:
                print(p)
                print('------------------------------')

