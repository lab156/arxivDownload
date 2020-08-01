from lxml import etree
import tarfile
import sys
import collections as coll
import time
import random
import parsing_xml as px
from functools import partial


def article_name_dict(tar_obj):
    """
    tar_obj is an object produced with tarfile.open(...)
    Returns a default dict with key article name ex. math.9203203
    and value: ['9203_001/math.9203203/sl2z.xml']
    """
    article_dict = coll.defaultdict(list)
    for pathname in tar_obj.getnames():
        dirname = pathname.split('/')[1]
        article_dict[dirname].append(pathname)
    return article_dict

def tar_iter(tarpath, patt):
    """
    returns and iterator to the file objects of a tar zip that have a certain
    pattern in their names
    """
    retries = 0
    while retries < 10:
        retries += 1
        try:
            with tarfile.open(tarpath) as tar_file:
                for fname in filter(lambda n: patt in n, tar_file.getnames()):
                    yield (fname, tar_file.extractfile(fname))
            break
        except tarfile.ReadError as ee:
            wait_delay = random.randint(5,15)
            print(ee, "while opening tarfile: %s, waiting for %s"%(tarpath, wait_delay))
            time.sleep(wait_delay)

def tar(tarpath, *args):
    """
    tarpath: Is the address of a processed by LaTeXML .tar.gz file
    args may be: 
          int: return the i-th file
          str: return the first match of the names in the tarfile
    """
    with tarfile.open(tarpath) as tar_file:
        if args:
            if isinstance(args[0], int):
                contains = lambda ftype,fname: ftype in fname
                xmlname = list(filter(partial(contains, '.xml'), tar_file.getnames()))[args[0]]
                logname = list(filter(partial(contains, '.txt'), tar_file.getnames()))[args[0]]
            elif isinstance(args[0], str): 
                def contains(ftype,fname): 
                    return (args[0] in fname and ftype in fname)
                try:
                    xmlname = next(filter(partial(contains, '.xml'), tar_file.getnames()))
                    xml_xtract = px.DefinitionsXML(tar_file.extractfile(xmlname))
                except StopIteration:
                    print(f"No xml results for {args[0]} in {tarpath}")
                    xmlname = None
                try:
                    logname = next(filter(partial(contains, '.txt'), tar_file.getnames()))
                    log_xtract = tar_file.extractfile(logname).read().decode('utf8')
                except StopIteration:
                    print(f"No txt results for {args[0]} in {tarpath}")
                    logname = None
        else:
            xmlname = tar_file.getnames()[0]
        
    return (log_xtract, xml_xtract)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='parsing xml commandline script')

    parser.add_argument('tarname', type=str, nargs=1,
            help='tar.gz file name to peep into. ex. math.9213435.tar.gz or \
            1803_343343.tar.gz')

    parser.add_argument('-p', '--pattern', help='pattern in the name of the article to \
            decompress',
            type=str)

    parser.add_argument('-c', '--commentary', action='store_const', const=True, help='print the commentary file')
    parser.add_argument('-e', '--errors', action='store_const', const=True, help='print the commentary file')


    args = parser.parse_args(sys.argv[1:])

    if args.pattern:
        patt = args.pattern
    else:
        patt = ''

    with tarfile.open(args.tarname[0]) as tar_file:
        article_dict = article_name_dict(tar_file)
        for name,val in filter(lambda n: patt in n[0], article_dict.items()):
            if args.commentary:
                fobj = tar_file.extractfile(next(filter(lambda s: 'comment' in s, val)))
                print(fobj.read().decode('utf-8'))
            elif args.errors:
                fobj = tar_file.extractfile(next(filter(lambda s: 'errors_mes' in s, val)))
                print(fobj.read().decode('utf-8'))
            else:
                fobj = tar_file.extractfile(next(filter(lambda s: '.xml' in s, val)))
                the_tree = etree.parse(fobj)
                print(etree.tostring(the_tree, pretty_print=True).decode('utf-8'))
