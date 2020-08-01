# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from lxml import etree
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import sys
import os
sys.path.insert(0,'arxiv.py/')
import arxiv
import databases.create_db_define_models as cre

with open('/mnt/arXiv_src/src/arXiv_src_manifest.xml', 'r') as f:
    mani = etree.parse(f)
    
# Get list of filenames with xpath
fname = mani.xpath('.//filename')
len(fname)
fname_set = set([f.text for f in fname])
# -

# %load_ext autoreload
# %autoreload 2
import process as pro

fname_lst = set([f.text for f in fname])
sorted(fname_lst)[1400:]

# +
# Connect to the database
database = 'sqlite:///../db_test.db'
downloaded_tars = '/mnt/arXiv_src/downloaded_log.csv'
eng = sa.create_engine(database, echo=False)
eng.connect()
SMaker = sessionmaker(bind=eng)
sess = SMaker()
# don't need this unless creating database from scratch
cre.ManifestTarFile.metadata.create_all(eng)
cre.Article.metadata.create_all(eng)

# Get tarfile set of names
q = sess.query(cre.ManifestTarFile)
filename_set = set([f.filename for f in q.all()])

# Get set of files that have not been fetched for metadata
downloaded_df = pd.read_csv(downloaded_tars)
downloaded_set = set(downloaded_df.filename)
# for some reason src/arXiv_src_manifest.xml is showing up
diff_set = downloaded_set.difference(filename_set)\
                         .difference(set(['src/arXiv_src_manifest.xml']))

len(diff_set)
# -

# Get the name of the file that appears in the manifest file
# Ex. src/arXiv_src_sdf23424.tar
manifest_name = '/'.join(x.tar_path.split('/')[-2:])
MM = mani.xpath('.//*[filename[text()="%s"]]'%manifest_name)[0]
line = parse_element(MM)
f = cre.ManifestTarFile(**line)
sess.add(f)
sess.commit()
x.save_articles_to_db('sqlite:///../db_test.db')

# +
root = mani.getroot()
def parse_element(elem):
    """ return dictionary with manifest metadata """
    return_dict = {}
    for e in elem:
        return_dict[e.tag] = e.text
    return return_dict
def parse_root(root):
        return [parse_element(child) for child in iter(root) if child.tag != 'timestamp']

filedf = pd.DataFrame(parse_root(root))
filedf[['num_items', 'size']] = filedf[['num_items', 'size']].astype(int)
filedf['filename'] = filedf['filename'].astype(str)
# -

prob = pro.Xtraction('tests/minitest4.tar')

prob.extract_tar('data/rm_me_0504_002', 'math')

prob.encoding_dict

mini.save_articles_to_db('sqlite:///tests/test.db')

for q in mini.query_results:
    print(type(q['tags']),"---", q['arxiv_primary_category'])
    print('')

mini.art_lst

eng = sa.create_engine('sqlite:///tests/test.db', echo=True)
eng.connect()
SMaker = sessionmaker(bind=eng)
sess = SMaker()
#cre.ManifestTarFile.metadata.create_all(eng)
#cre.Article.metadata.create_all(eng)

sess.query(cre.Article).join(cre.ManifestTarFile).filter(cre.ManifestTarFile.id == 3).all()

sess.query(cre.Article).filter(cre.Article.tarfile_id == 5).delete(synchronize_session='fetch')
sess.commit()
sess.query(cre.Article).filter(cre.Article.tarfile_id == 5).all()

pks = sess.query(cre.Article.pk).join(cre.ManifestTarFile).filter(cre.ManifestTarFile.filename.like("src/arXiv_src_133%"))
pks.all()

# + jupyter={"outputs_hidden": true}
# POPULATE THE MANIFEST TABLE (Careful)
#for index, line in filedf.iterrows():
    #line = filedf.iloc[50*k]
    f = cre.ManifestTarFile(**line)
    sess.add(f)
    sess.commit()
# -

D = arxiv.query(id_list=['1901.009'])[0]
g = cre.new_article_register(D,2)
#sess.add(g)
#sess.commit()

q = sess.query(cre.ManifestTarFile)
resu = q.filter(cre.ManifestTarFile.num_items > 3000)
resu_set = set([f.filename for f in q.all()])

diff_set = fname_set.difference(resu_set)

diff_set.pop()

len(diff_set)

while diff_set:
    diff_set.pop()
