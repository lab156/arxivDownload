from lxml import etree
from bz2 import BZ2File
import re
from unwiki import unwiki

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
    regex_def = re.findall(reg_expr, text.text)
    output_list =[]
    for r in regex_def:
        defin_section = unwiki.loads(r[2])
        out_dict = {
                'title': title,
        'section': r[1],
        'definition': defin_section,
        'matches': title.lower() in defin_section.lower(), }
        output_list.append(out_dict)
    return output_list

def fast_iter(xml_file, out_file=None, sentinel=None):
    '''
    xml_file : is a readable file object
    sentinel : is an int of the max number of pages to check
    '''
    sep_string = ' -#-%- '
    parser = etree.iterparse(xml_file, tag=ns+'page')
    senti = 0
    for event,elem in parser:
        res_lst = search_defs(elem, namespace=nsp)
        for r in res_lst:
            if r['matches']:
                print('\033[1m' + r['title'] + '\033[0m',
                sep_string , r['section'], sep_string, r['definition'])
                if out_file:
                    print(r['title'] ,
                sep_string , r['section'], sep_string, r['definition'])
        elem.clear()
        #text.clear()
        #title.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        #while text.getprevious() is not None:
        #    del text.getparent()[0]
        #while title.getprevious() is not None:
        #    del title.getparent()[0]

        if sentinel:
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
    parser = argparse.ArgumentParser(
            description='Parse the wikipedia dump file')
    parser.add_argument('path', help='wikipedia .bz2 file')
    parser.add_argument('-s', '--sentinel',
            help='file to write the logs', default=None)
    parser.add_argument('-o', '--outfile', 
            help='file to write the definitions', default=None)
    args = parser.parse_args()
    with BZ2File(args.path) as xml_file:
        if args.outfile:
            with open(args.outfile, 'a') as out_file_f:
                fast_iter(xml_file, out_file=out_file_f,
                        sentinel=int(args.sentinel))
        else:
            fast_iter(xml_file, sentinel=int(args.sentinel))
