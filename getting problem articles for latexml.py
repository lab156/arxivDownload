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
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import sys
import os
sys.path.insert(0,'arxiv.py/')
import arxiv
import databases.create_db_define_models as cre
import re

def fun1(name):
    '''
   takes the string: ./error/1703/1703.01344.html\n
   returns 1703.01344
    '''
    res = re.match(r'.*/([0-9]{4}\.[0-9]{5})\.html\n', name)
    return res.group(1)

def fun2(name):
    '''
    takes the string: 
    '''
    res = re.match(r'.*/([0-9]{4}\.[0-9]{5})', name)
    return res.group(1)
    
fun2('http://arxiv.org/abs/1703.01333v2')
# -

# using a list of files in the arxxmllib dataset of processed articles
# find . -name '1703.013*' > ~/find_start_1703.013.txt
with open('../find_start_1703.013.txt', 'r') as find_file:
    art_lst = find_file.readlines()

# Connect to the metadata database in search of articles that contain the string: %1703.013%
database = 'sqlite:///../arxiv2.db'
eng = sa.create_engine(database, echo=False)
eng.connect()
SMaker = sessionmaker(bind=eng)
sess = SMaker()
#sess.query(cre.Article.id).filter(cre.Article.tags.startswith("[{'term': 'math.DG'")).all()
q_lst = sess.query(cre.Article.id).filter(cre.Article.id.like("%1703.013%")).all()

arxm_set = set([fun2(a) for a in art_lst])
all_set = set([fun2(a[0]) for a in q_lst])
prob_set = all_set.difference(arxm_set)

for nm in prob_set:
    q_str = "%{}%".format(nm)
    print(sess.query(cre.Article.tarfile_id).filter(cre.Article.id.like(q_str)).all())

prob_set


