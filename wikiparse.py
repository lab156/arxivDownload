from lxml import etree
from bz2 import BZ2File
import re

reg_expr = re.compile('(?P<nequals>=+)([\w\s]*?definition) *(?P=nequals)\n(.+?)\n\n=+', re.I|re.DOTALL)
nsp = {'wiki':'http://www.mediawiki.org/xml/export-0.10/'}
ns = '{http://www.mediawiki.org/xml/export-0.10/}'

def search_defs(page_tree, namespace={'wiki':''}):
    '''
    Given a parsed xml tree (ex. the output of lxml.etree.parse(xml_file))
    returns a LIST of all the definitions sections splitted as a dictionary
    in the following format:
        {title: the title of the page,
        section: the name of the definition section,
        definition: the text of all the definition section
        }
        '''
    text = page_tree.find('wiki:revision/wiki:text', namespaces=namespace)
    title = page_tree.find('wiki:title', namespaces=namespace).text
    # takes care of articles with several definition sections
    try:
        regex_def = re.findall(reg_expr, text.text)
    except TypeError:
        print('Article %s has no text'%title)
        return []

    output_list =[]
    for r in regex_def:
        defin_section = r[2]
        out_dict = {
                'title': title,
        'section': r[1],
        'definition': defin_section,
        'matches': title.lower().strip() in defin_section.lower(), }
        output_list.append(out_dict)
    return output_list

def fast_iter(xml_file, out_file=None, sentinel=None):
    '''
    xml_file : is a readable file object
    sentinel : is an int of the max number of pages to check
    '''
    ns = '{http://www.mediawiki.org/xml/export-0.10/}'
    nsp = {'wiki':'http://www.mediawiki.org/xml/export-0.10/'}
    parser = etree.iterparse(xml_file, tag=ns+'page')
    reg_expr = re.compile('=+ ([\w ]*definition) =+\n(.+?)\n\n', re.I|re.DOTALL)
    senti = 0
    for event,elem in parser:
        text = elem.find('wiki:revision/wiki:text', namespaces=nsp)
        title = elem.find(ns+'title')
        try:
            regex_def = re.findall(reg_expr, text.text)
        except TypeError:
            print('Article with title %s has no text :('%title.text)
        if regex_def:
            if title.text.lower() in regex_def[0][1].lower():
                print('\033[1m' + title.text + '\033[0m', ' --- ', regex_def[0][0], ' --- ', regex_def[0][1])
                if out_file:
                    print(r['title'] ,
                sep_string , r['section'], 
                sep_string, r['definition'], file=out_file)
        elem.clear()
        text.clear()
        title.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        while text.getprevious() is not None:
            del text.getparent()[0]
        while title.getprevious() is not None:
            del title.getparent()[0]

        if sentinel > 0:
            if senti > sentinel:
                break
            senti += 1


if __name__ == '__main__':
    '''
    Usage:
        python wikiparse.py ../enwiki-20190201-pages-articles-multistream1.xml-p10p30302.bz2 -s 650
python wikiparse.py ../enwiki-20190201-pages-articles-multistream1.xml-p10p30302.bz2 -s 1500 -o ../test_def.txt
    '''

    import argparse
    parser = argparse.ArgumentParser( description='Parse the wikipedia dump file')
    parser.add_argument('path', help='wikipedia .bz2 file')
    parser.add_argument('-s', '--sentinel',
            help='file to write the logs', default=0)
    parser.add_argument('-o', '--outfile', 
            help='file to write the definitions', default=None)
    args = parser.parse_args()
    with BZ2File(args.path) as xml_file:
        if args.outfile:
            with open(args.outfile, 'a') as out_file_f:
                fast_iter(xml_file, out_file=out_file_f, sentinel=int(args.sentinel))
        else:
            fast_iter(xml_file, sentinel=int(args.sentinel))
