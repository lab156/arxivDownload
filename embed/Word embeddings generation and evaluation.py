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
from scipy.cluster.vq import kmeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap

# %load_ext autoreload
# %autoreload 2
from report_lengths import generate
# -

------- Do not run -----
# This strips all the text from the xml articles and saves to text file
for math_year in ['math12', 'math13', 'math14','math16','math17','math18','math19', 'math20']:
    #math_year = 'math97'
    for art_path in tqdm(glob.glob('/mnt/promath/{}/*.tar.gz'.format(math_year))):
        art_str = ""
        for name, fobj in peep.tar_iter(art_path, '.xml'):
            try:
                article = px.DefinitionsXML(fobj)
                art_str = " ".join([article.recutext(a) for a in article.para_list()])
            except ValueError as ee:
                #print(ee, f"file {name} produced an error")
                art_str = " "
            with open('../data/clean_text/{}'.format(math_year),'a') as art_fobj:
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

# +
# %%time
# Connect to the database
database = 'sqlite:///../../arxiv3.db'
eng = sa.create_engine(database, echo=False)
eng.connect().execute('pragma case_sensitive_like=OFF;')
SMaker = sessionmaker(bind=eng)
sess = SMaker()

def qq(art_str, database = 'sqlite:///../../arxiv3.db'):
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
# -

## Grab the Glossary data
dfndum_set = set()
new_dfndum_lst = [0]
tot_dfndum_lst = [0]
rep_ratio = []
term_cnt = Counter()
term_dict_cnt = defaultdict(Counter)
perc_array = np.array([])
for xml_path in tqdm(glob.glob('/mnt/glossary/v2/math*/*.xml.gz')):
    gtree = etree.parse(xml_path).getroot()
    for art in gtree.iter(tag='article'):
        d_lst = [d.text.lower() for d in art.findall('.//dfndum')]
        dfndum_set.update(d_lst)
        term_cnt.update(d_lst)
        new_dfndum_lst.append(len(dfndum_set))
        tot_dfndum_lst.append(tot_dfndum_lst[-1] + len(d_lst))
        rep_ratio.append(tot_dfndum_lst[-1]/len(dfndum_set))
        try:
            arxiv_class = qq(art.attrib['name'].split('/')[1])
            #print(f"Found arxiv class {arxiv_class}")
            for D in d_lst:
                term_dict_cnt[D].update([arxiv_class])
        except StopIteration:
            pass

# The 15 most common words are
term_cnt.most_common()[:15]

s = 304
Term = term_cnt.most_common()[s][0]
print(f'The term: {Term} appears in articles tagged:')
term_dict_cnt[Term]

# Decode word2vec .bin file
with open('/mnt/embeddings/model14-14_12-08/vectors.bin', 'rb') as mfobj:
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

common_term = term_cnt.most_common()[200][0].lower().replace(' ', '_')
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

# Show some typical elements
print(list(veryAG.keys())[:15])
print(list(veryDG.keys())[:15])

ag_lst = [v[0] for v in veryAG.values()][:500]
dg_lst = [v[0] for v in veryDG.values()][:500]
tot_vec = np.stack(ag_lst + dg_lst, axis=0)
labels_vec = len(ag_lst)*['math.AG'] + len(dg_lst)*['math.DG']
tsne2 = TSNEVisualizer(labels=['math.AG','math.DG'])
tsne2.fit(tot_vec, labels_vec)
tsne2.show(figsize=100)

tsne1 = TSNE()
tot_vec = np.stack(ag_lst + dg_lst, axis=0)
means = kmeans(tot_vec, 10)
tot_vec = np.concatenate([tot_vec, means[0]], axis=0)
labels_vec = len(ag_lst)*['math.AG'] + len(dg_lst)*['math.DG'] + 2*['center']
tran_vec = tsne1.fit_transform(tot_vec, labels_vec)
x,y =  list(zip(tran_vec.transpose()))
plt.figure(figsize=[7,7])
plt.scatter(x[0][:500],y[0][:500])
plt.scatter(x[0][500:1000], y[0][500:1000], color='green')
plt.scatter(x[0][1000:], y[0][1000:], color='red')
plt.savefig('/home/luis/tsne_ag_dg.png')
plt.show()
for k,center in enumerate(means[0]):
    print(f"----------- Center {k} nearest neighbors ------------")
    for word,dist in nearest(center, n_near=7):
        print(word, "{0:3.2f}".format(dist))

umap1 = umap.UMAP()
tot_vec = np.stack(ag_lst + dg_lst, axis=0)
means = kmeans(tot_vec, 3)
tot_vec = np.concatenate([tot_vec, means[0]], axis=0)
labels_vec = len(ag_lst)*['math.AG'] + len(dg_lst)*['math.DG'] + len(means[0])*['center']
tran_vec = umap1.fit_transform(tot_vec, labels_vec)
x,y =  list(zip(tran_vec.transpose()))
plt.figure(figsize=[7,7])
plt.scatter(x[0][:500],y[0][:500])
plt.scatter(x[0][500:1000], y[0][500:1000], color='green')
plt.scatter(x[0][1000:], y[0][1000:], color='red')
plt.show()

# +
cos_dist = lambda x, y: np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)
unit_embed = {w: v/np.linalg.norm(v) for w,v in embed.items()}

def nearest(word_vec, n_near=10):
    dist_dict = {}
    unit_word_vec = word_vec/np.linalg.norm(word_vec)
    for w, v in unit_embed.items():
        #dist_dict[w] = cos_dist(v, word_vec)
        dist_dict[w] = unit_word_vec.dot(v)
    return sorted(dist_dict.items(), key=lambda pair: pair[1], reverse=True)[:n_near]


# +
#topic,cap = ('math.GN',3) # General Topology
#topic,cap = ('math.GT', 15) 
#topic,cap = ('math.AT', 10) #poor results
#topic,cap = ('math.DG', 10) 
#topic,cap = ('math.LO', 5) 
#topic,cap = ('math.DS', 15) 
#topic,cap = ('math.PR', 15) # very "graphy" center
#topic,cap = ('math.NT', 15) 
#topic,cap = ('math.FA', 15) 
#topic,cap = ('math.GM', 2) 
topic,cap = ('math.OC', 5) 


veryTop = {}
color_dict = {}
for Term_pair in tqdm(term_cnt.most_common()):
    Term = Term_pair[0]
    if term_dict_cnt[Term][topic] > cap:
        emb_term = Term.lower().replace(' ', '_')
        embed_vec = embed.get(emb_term, None)
        if embed_vec is not None: 
            veryTop[Term] = embed_vec
            color_dict[Term] = float(term_dict_cnt[Term][topic])/sum(term_dict_cnt[Term].values())
# -

# %%time
tsne1 = TSNE()
term_lst = list(veryTop.keys())
tot_vec = np.stack([veryTop[t] for t in term_lst], axis=0)
n_centers = 10
colors = [3*[color_dict[t]*0.8] for t in term_lst] + n_centers*[[1,0,0]]
means = kmeans(tot_vec, n_centers)
tot_vec = np.concatenate([tot_vec, means[0]], axis=0)
#labels_vec = len(ag_lst)*['math.AG'] + len(dg_lst)*['math.DG'] + 2*['center']
tran_vec = tsne1.fit_transform(tot_vec)
x,y =  list(zip(tran_vec.transpose()))
plt.figure(figsize=[7,7])
plt.scatter(x[0],y[0],c=colors)
plt.show()
for k,center in enumerate(means[0]):
    print(f"----------- Center {k} nearest neighbors ------------")
    near_lst = nearest(center, n_near=7)
    cnt_list = []
    for word,dist in near_lst:
        print(word, "{0:3.2f}".format(dist),
              sum(term_dict_cnt[word].values()))
        cnt_list.append( sum(term_dict_cnt[word].values()))
    print( '----- ',max(cnt_list))

n_average = 5 # Number of samples to average out
dist_lst = []
for n_centers in tqdm(range(2,20)):
    mean_dist = 0
    for _ in range(n_average):
        mean_dist += kmeans(tot_vec, n_centers)[1]
    dist_lst.append(mean_dist/n_average)
plt.plot(dist_lst)
plt.show()

term_dict_cnt['markov chain']
