# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
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
from collections import defaultdict
from process import Xtraction

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
database = 'sqlite:////mnt/databases/arxiv2.db'
eng = sa.create_engine(database, echo=False)
eng.connect()
SMaker = sessionmaker(bind=eng)
sess = SMaker()
#sess.query(cre.Article.id).filter(cre.Article.tags.startswith("[{'term': 'math.DG'")).all()
q_lst = sess.query(cre.Article.id).filter(cre.Article.id.like("%1703.013%")).all()

arxm_set = set([fun2(a) for a in art_lst])
all_set = set([fun2(a[0]) for a in q_lst])
prob_set = all_set.difference(arxm_set)

# Get the tar_id of each problem files
id_dict = defaultdict(list)
for nm in prob_set:
    q_str = "%{}%".format(nm)
    query_resu = sess.query(cre.Article.tarfile_id, cre.Article.id).filter(cre.Article.id.like(q_str)).all()[0]
    tar_resu = sess.query(cre.ManifestTarFile.filename).filter(cre.ManifestTarFile.id == int(query_resu[0])).all()
    id_dict[tar_resu[0][0]].append(query_resu[1]) 
    print(tar_resu)

# Extract the files
for tar in id_dict:
    x = Xtraction('/mnt/arXiv_src/' + tar, db='sqlite:////mnt/databases/arxiv2.db')
    x.extract_tar('../rm_me_problem_file', article_name='^1703/1703\.013.*')

id_dict

# ### This is the problem id dict:
# defaultdict(list,
#             {'src/arXiv_src_1703_003.tar': ['http://arxiv.org/abs/1703.01327v2',
#               'http://arxiv.org/abs/1703.01306v2',
#               'http://arxiv.org/abs/1703.01331v1',
#               'http://arxiv.org/abs/1703.01314v1',
#               'http://arxiv.org/abs/1703.01305v2'],
#              'src/arXiv_src_1703_004.tar': ['http://arxiv.org/abs/1703.01399v1',
#               'http://arxiv.org/abs/1703.01375v2',
#               'http://arxiv.org/abs/1703.01396v2',
#               'http://arxiv.org/abs/1703.01373v1',
#               'http://arxiv.org/abs/1703.01379v3',
#               'http://arxiv.org/abs/1703.01356v4',
#               'http://arxiv.org/abs/1703.01357v3',
#               'http://arxiv.org/abs/1703.01397v1',
#               'http://arxiv.org/abs/1703.01395v1',
#               'http://arxiv.org/abs/1703.01376v4',
#               'http://arxiv.org/abs/1703.01352v1',
#               'http://arxiv.org/abs/1703.01354v2',
#               'http://arxiv.org/abs/1703.01378v1']})


