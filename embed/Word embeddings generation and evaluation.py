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
import struct as st
from itertools import islice
import numpy as np

# %load_ext autoreload
# %autoreload 2
from report_lengths import generate
# -

for art_path in tqdm(glob.glob('/mnt/promath/math15/*.tar.gz')):
    art_str = ""
    for name, fobj in peep.tar_iter(art_path, '.xml'):
        try:
            article = px.DefinitionsXML(fobj)
            art_str = " ".join([article.recutext(a) for a in article.para_list()])
        except ValueError as ee:
            #print(ee, f"file {name} produced an error")
            art_str = " "
        with open('../data/math15','a') as art_fobj:
            print(art_str, file=art_fobj)
        #print(f"Saved art {name} from {art_path}")

dfndum_set = set()
new_dfndum_lst = [0]
tot_dfndum_lst = [0]
rep_ratio = []
term_cnt = Counter()
perc_array = np.array([])
for xml_path in tqdm(glob.glob('/mnt/glossary/v1.1/math19/*.xml.gz')):
    gtree = etree.parse(xml_path).getroot()
    for art in gtree.iter(tag='article'):
        d_lst = [d.text for d in art.findall('.//dfndum')]
        dfndum_set.update(d_lst)
        term_cnt.update(d_lst)
        new_dfndum_lst.append(len(dfndum_set))
        tot_dfndum_lst.append(tot_dfndum_lst[-1] + len(d_lst))
        rep_ratio.append(tot_dfndum_lst[-1]/len(dfndum_set))


s = 100
term_cnt.most_common()[s:s+10]

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
    #print(res.id)
    return lst[0]
qq('1912')
# -

with open('../../word2vec/math19-vectors-phrase.bin', 'rb') as mfobj:
    m = mfobj.read()
    #print(m[0].decode('utf8'))
    #s = st.Struct('ii')
    #m_it = m.__iter__()
    head_dims = st.unpack('<11s', m[:11])
    n_vocab, n_dim = map(int,head_dims[0].strip().split())
    print(f"Vocabulary size: {n_vocab} and dimension of embed: {n_dim}")
    embed = {}
    #[next(m_it) for _ in range(11)]
    cnt = 11
    for line_cnt in tqdm(range(n_vocab)):
        word = ''
        while True:
            next_char = st.unpack('<1s', m[cnt:cnt+1])[0].decode('utf8')
            cnt += 1
            if next_char == ' ':
                break
            else:
                word += next_char
        #print(word)
        vec = np.zeros(200)
        for k in range(200):
            vec[k] = st.unpack('<f', m[cnt:cnt+4])[0]
            cnt += 4
        assert st.unpack('<1s', m[cnt:cnt+1])[0] == b'\n'
        cnt +=1
        embed[word] = vec

embed['integer']


