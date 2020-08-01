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
from collections import Counter, defaultdict
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import databases.create_db_define_models as cre
import re
import struct as st
from itertools import islice
import numpy as np
from yellowbrick.text import TSNEVisualizer

# %load_ext autoreload
# %autoreload 2
from report_lengths import generate
# -

------- Do not run -----
# This strips all the text from the xml articles and saves to text file
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


# +
# %%time
# Connect to the database
#database = 'sqlite:///../../arxiv2.db'
#eng = sa.create_engine(database, echo=False)
#eng.connect().execute('pragma case_sensitive_like=OFF;')
#SMaker = sessionmaker(bind=eng)
#sess = SMaker()

# Get tarfile set of names
def qq(art_str):
    q = sess.query(cre.Article)
    res = q.filter(cre.Article.id.like("http://arxiv.org/abs/"+ art_str + "%")).first()
    try:
        lst = eval(res.tags)[0]['term']
    except AttributeError:
        lst = None
    print(res.id)
    return lst
qq('1512.09109')


# -

# %%time
def qq(art_str, database = 'sqlite:///../../arxiv2_test_ind.db'):
    # Connect to the database
    eng = sa.create_engine(database, echo=False)
    with eng.connect() as con:
        con.execute('pragma case_sensitive_like=OFF;')
        #resu = con.execute('select tags from articles where id like "http://arxiv.org/abs/1512.09109%;"')
        #resu = con.execute('select tags from articles where id glob "http://arxiv.org/abs/1512.09109*";')
        q_str = '''select tags from articles where id 
        between "http://arxiv.org/abs/{0}" and "http://arxiv.org/abs/{0}{{";'''.format(art_str)
        resu = next(con.execute(q_str))
    #resu format: ("[{'term': 'math.RT', 'scheme': 'http://arxiv.org/schemas/atom', 'label': None}]",)
    res_dict = eval(resu[0])
    return res_dict[0]['term']
qq('1512.09109')    

## Grab the Glossary data
dfndum_set = set()
new_dfndum_lst = [0]
tot_dfndum_lst = [0]
rep_ratio = []
term_cnt = Counter()
term_dict_cnt = defaultdict(Counter)
perc_array = np.array([])
for xml_path in tqdm(glob.glob('/mnt/glossary/v1.1/math15/*.xml.gz')):
    gtree = etree.parse(xml_path).getroot()
    for art in gtree.iter(tag='article'):
        d_lst = [d.text for d in art.findall('.//dfndum')]
        dfndum_set.update(d_lst)
        term_cnt.update(d_lst)
        new_dfndum_lst.append(len(dfndum_set))
        tot_dfndum_lst.append(tot_dfndum_lst[-1] + len(d_lst))
        rep_ratio.append(tot_dfndum_lst[-1]/len(dfndum_set))
        arxiv_class = qq(art.attrib['name'].split('/')[1])
        #print(f"Found arxiv class {arxiv_class}")
        for D in d_lst:
            term_dict_cnt[D].update([arxiv_class])

# + jupyter={"outputs_hidden": true}
term_cnt.most_common()[25:]
# -

s = 300
Term = term_cnt.most_common()[s]
print(f'The term: {Term} appears in articles tagged:')
term_dict_cnt[Term]

# Decode word2vec .bin file
with open('../../word2vec/math15-vectors-phrase.bin', 'rb') as mfobj:
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

common_term = term_cnt.most_common()[2001][0].lower().replace(' ', '_')
print(f" The term is: {common_term}")
embed.get(common_term, None)[:10]

# Create a dict of "very AG" vectors
veryAG = {}
veryDG = {}
for Term_pair in tqdm(term_cnt.most_common()):
    Term = Term_pair[0]
    if term_dict_cnt[Term]['math.AG'] > 4*term_dict_cnt[Term]['math.DG']:
        embed_name = Term.lower().replace(' ', '_')
        try:
            embed_vec = embed.get(embed_name, None)
            tot = float(term_dict_cnt[Term]['math.AG'] + term_dict_cnt[Term]['math.DG'])
            color_intensity = (term_dict_cnt[Term]['math.AG']/tot, term_dict_cnt[Term]['math.DG']/tot)
            if embed_vec is not None:
                veryAG[Term] = (embed_vec, color_intensity)
        except TypeError:
            pass
    if term_dict_cnt[Term]['math.DG'] > 4*term_dict_cnt[Term]['math.AG']:
        embed_name = Term.lower().replace(' ', '_')
        try:
            embed_vec = embed.get(embed_name, None)
            tot = float(term_dict_cnt[Term]['math.AG'] + term_dict_cnt[Term]['math.DG'])
            color_intensity = (term_dict_cnt[Term]['math.AG']/tot, term_dict_cnt[Term]['math.DG']/tot)
            if embed_vec is not None:
                veryDG[Term] = (embed_vec, color_intensity)
        except TypeError:
            pass

print(list(veryAG.keys())[:15])
print(list(veryDG.keys())[:15])
veryAG['graph']

ag_vec = [v[0] for v in veryAG.values()]
len(ag_vec[0])

ag_lst = [v[0] for v in veryAG.values()][:500]
dg_lst = [v[0] for v in veryDG.values()][:500]
tot_vec = np.stack(ag_lst + dg_lst, axis=0)
labels_vec = len(ag_lst)*['math.AG'] + len(dg_lst)*['math.DG']
tsne2 = TSNEVisualizer(labels=['math.AG','math.DG'])
tsne2.fit(tot_vec, labels_vec)
tsne2.poof(figsize=100)


