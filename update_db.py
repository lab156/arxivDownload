from lxml import etree
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import sys
sys.path.insert(0,'arxiv.py/')
import arxiv
import databases.create_db_define_models as cre
import process as pro
import os

def parse_element(elem):
    """ return dictionary with manifest metadata """
    return_dict = {}
    for e in elem:
        return_dict[e.tag] = e.text
    return return_dict
def parse_root(root):
        return [parse_element(child) for child in iter(root) if child.tag != 'timestamp']

if __name__ == "__main__":
    '''
        * USAGE: python update_db.py DATABASE MANIFEST.xml tar_src_path [--log ]
    '''
    import argparse
    parser = argparse.ArgumentParser(description='parsing xml commandline script')
    parser.add_argument('database', type=str,
            help='sqlite database with articles and manifest tables ex. ../db_test.db')
    parser.add_argument('manifest', type=str,
            help='''manifest xml file assumming that all the files have been
            downloaded in the src directory ex.  ../arXiv_src_manifest_Oct_2019.xml''')
    parser.add_argument('tar_path', type=str,
            help='path of the src file where the arxiv tars have been downloaded ex. /mnt/arXiv_src/src')

    args = parser.parse_args(sys.argv[1:])
    print(args.manifest)

    with open(args.manifest, 'r') as f:
        mani = etree.parse(f)
    database = 'sqlite:///' + args.database
    # I am assuming downloaded files = manifest
    #downloaded_tars = '/mnt/arXiv_src/downloaded_log.csv'
    eng = sa.create_engine(database, echo=False)
    eng.connect()
    SMaker = sessionmaker(bind=eng)
    sess = SMaker()
    # don't need this unless creating database from scratch
    #cre.ManifestTarFile.metadata.create_all(eng)
    #cre.Article.metadata.create_all(eng)

    # Get tarfile set of names
    q = sess.query(cre.ManifestTarFile)
    filename_set = set([f.filename for f in q.all()])

    # Get set of files that have not been fetched for metadata
    #downloaded_df = pd.read_csv(downloaded_tars)
    fname = mani.xpath('.//filename')
    downloaded_set = set([f.text for f in fname]) #set(downloaded_df.filename)
    # for some reason src/arXiv_src_manifest.xml is showing up
    diff_set = downloaded_set.difference(filename_set)\
                             .difference(set(['src/arXiv_src_manifest.xml']))

    # create the xtract object
    while diff_set:
        x = pro.Xtraction(os.path.join(args.tar_path, diff_set.pop()))

        # Get the name of the file that appears in the manifest file
        # Ex. src/arXiv_src_sdf23424.tar
        manifest_name = '/'.join(x.tar_path.split('/')[-2:])
        MM = mani.xpath('.//*[filename[text()="%s"]]'%manifest_name)[0]
        line = parse_element(MM)
        f = cre.ManifestTarFile(**line)
        sess.add(f)
        sess.commit()
        x.save_articles_to_db(database)

