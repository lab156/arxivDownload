from lxml import etree
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import os
import re
import sys
import parsing_xml as px

eng = sa.create_engine('sqlite:///../arxiv1.db')
eng.connect()
SMaker = sa.orm.sessionmaker(bind=eng)
sess = SMaker()

def create_dict():
    xml = etree.parse('/home/luis/media_home/xml_file.xml')
    art_dict = {}
    for k, x in enumerate(xml.iter('article')):
        if x.get('searched') == 'True':
            art_dict[x.find('id').text] = x.find('location').text
    return art_dict

def query():
    return sess.execute('''SELECT id FROM articles
           where tags LIKE  '[{''term'': ''math.AG''%' and
           updated_parsed BETWEEN date('2016-01-01')  and date('2018-11-20');''')

loc_path = '/home/luis/media_home/'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='parsing xml commandline script')
    parser.add_argument('file_names', type=str, nargs=2,
            help='filenames of the the real definition and nondefs respectively')
    parser.add_argument('-l', '--logfile', help='file to write the logs', type=str)
    args = parser.parse_args(sys.argv[1:])

    print('  Creating the dictionary                                                      ', end='\r')
    art_dict = create_dict()

    print('  Querying                                                                     ', end='\r')
    qq = query()

    with open(args.file_names[0], 'a') as real_f, open(args.file_names[1], 'a') as nondefs_f:
        for l in qq:
            nonlocal_path = art_dict.get(l[0])
            if nonlocal_path:
                prepath = re.sub('^/mnt/', '', nonlocal_path)
                print('file: %s                                                            '%prepath, end='\r')
                local_path = os.path.join(loc_path, prepath)
                try:
                    xml = px.DefinitionsXML(local_path)
                    tdict = xml.get_def_sample_text_with()
                    for s in tdict['real']:
                        real_f.write(s + '\n')
                    for s in tdict['nondef']:
                        nondefs_f.write(s + '\n')
                except ValueError:
                    print('error  parsing file %s'%local_path)
            else:
                print('Did not found: %s                                                  '%l[0], end='\r')
