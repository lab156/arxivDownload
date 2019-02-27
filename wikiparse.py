from lxml import etree
from bz2 import BZ2File
import re

def fast_iter(xml_file, sentinel=None):
    '''
    xml_file : is a readable file object
    sentinel : is an int of the max number of pages to check
    '''
    ns = '{http://www.mediawiki.org/xml/export-0.10/}'
    nsp = {'wiki':'http://www.mediawiki.org/xml/export-0.10/'}
    parser = etree.iterparse(xml_file, tag=ns+'page')
    reg_expr = re.compile('== .*definition ==', re.I)
    senti = 0
    for event,elem in parser:
        text = elem.find('wiki:revision/wiki:text', namespaces=nsp)
        title = elem.find(ns+'title')
        if '== Definition ==' in text.text:
            print(title.text)
        elem.clear()
         text.clear()
        title.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        while text.getprevious() is not None:
            del text.getparent()[0]
        while title.getprevious() is not None:
            del title.getparent()[0]

        if sentinel:
            if senti > sentinel:
                break
            senti += 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser( description='Parse the wikipedia dump file')
    parser.add_argument('path', help='wikipedia .bz2 file')
    parser.add_argument('-s', '--sentinel', help='file to write the logs', default=None)
    args = parser.parse_args()
    with BZ2File(args.path) as xml_file:
        fast_iter(xml_file, sentinel=int(args.sentinel))
