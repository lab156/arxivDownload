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
import numpy as np
import sys

# IMPORT MODULES IN PARENT DIR
import sys, inspect, os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import parsing_xml as px
from extract import Definiendum
import peep_tar as peep
import glob
from tqdm import tqdm
from lxml import etree
from collections import Counter
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import databases.create_db_define_models as cre
import re

# %load_ext autoreload
# %autoreload 2
from report_lengths import generate
# -

for art_path in tqdm.tqdm(glob.glob('/mnt/promath/math19/*.tar.gz')):
    art_str = ""
    for name, fobj in peep.tar_iter(art_path, '.xml'):
        try:
            article = px.DefinitionsXML(fobj)
            art_str = " ".join([article.recutext(a) for a in article.para_list()])
        except ValueError as ee:
            #print(ee, f"file {name} produced an error")
            art_str = " "
        with open('../data/math19','a') as art_fobj:
            print(art_str, file=art_fobj)
        #print(f"Saved art {name} from {art_path}")

dfndum_set = set()
new_dfndum_lst = [0]
tot_dfndum_lst = [0]
rep_ratio = []
term_cnt = Counter()
perc_array = np.array([])
for xml_path in tqdm(glob.glob('/mnt/glossary/math19/*.xml.gz')):
    gtree = etree.parse(xml_path).getroot()
    for art in gtree.iter(tag='article'):
        d_lst = [d.text for d in art.findall('.//dfndum')]
        dfndum_set.update(d_lst)
        term_cnt.update(d_lst)
        new_dfndum_lst.append(len(dfndum_set))
        tot_dfndum_lst.append(tot_dfndum_lst[-1] + len(d_lst))
        rep_ratio.append(tot_dfndum_lst[-1]/len(dfndum_set))


term_cnt.most_common()[:10]

art.attrib["name"]

# +
# Connect to the database
database = 'sqlite:///../../arxiv2.db'
eng = sa.create_engine(database, echo=False)
eng.connect()
SMaker = sessionmaker(bind=eng)
sess = SMaker()

# Get tarfile set of names
def qq(art_str):
    q = sess.query(cre.Article)
    res = q.filter(cre.Article.id.like("%"+ art_str + "%")).first()
    lst = eval(res.tags)
    return lst[0]['term']
qq('1912\.0')
# -

mystring = "Hola como <s/> te va </s> _cite espero _inline_math_ que _item_ muy bien"
re.sub(r"</s>|_cite_|_item_", "", mystring)


