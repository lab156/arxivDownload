# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
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
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from matplotlib import cm
import umap
import scattertext as st
import random

from ripser import ripser
from ripser import Rips
from persim import plot_diagrams
import multiprocessing as mp

# %load_ext autoreload
# %autoreload 2
from embed_utils import generate, nearest, open_w2v 
from clean_and_token_text import normalize_text, token_phrases3

# + magic_args="echo skipping" language="script"
# ------- Do not run -----
# # This strips all the text from the xml articles and saves to text file
# for math_year in ['math12', 'math13', 'math14','math16','math17','math18','math19', 'math20']:
#     #math_year = 'math97'
#     for art_path in tqdm(glob.glob('/mnt/promath/{}/*.tar.gz'.format(math_year))):
#         art_str = ""
#         for name, fobj in peep.tar_iter(art_path, '.xml'):
#             try:
#                 article = px.DefinitionsXML(fobj)
#                 art_str = " ".join([article.recutext(a) for a in article.para_list()])
#             except ValueError as ee:
#                 #print(ee, f"file {name} produced an error")
#                 art_str = " "
#             with open('../data/clean_text/{}'.format(math_year),'a') as art_fobj:
#                 print(art_str, file=art_fobj)
#             #print(f"Saved art {name} from {art_path}")

# +
# %%time
# Connect to the database
database = 'sqlite:////media/hd1/databases/arxiv3.db'
eng = sa.create_engine(database, echo=False)
eng.connect().execute('pragma case_sensitive_like=OFF;')
SMaker = sessionmaker(bind=eng)
sess = SMaker()

def qq(art_str, database = database):
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
qq('1703.01352')    
# -

## Grab the Glossary data
dfndum_set = set()
new_dfndum_lst = [0]
tot_dfndum_lst = [0]
rep_ratio = []
term_cnt = Counter()
term_dict_cnt = defaultdict(Counter)
perc_array = np.array([])
stopiterations_lst = []
for xml_path in tqdm(glob.glob('/media/hd1/glossary/v3/math*/*.xml.gz')):
    gtree = etree.parse(xml_path).getroot()
    for art in gtree.iter(tag='article'):
        d_lst = [d.text.lower() for d in art.findall('.//dfndum')]
        dfndum_set.update(d_lst)
        term_cnt.update(d_lst)
        new_dfndum_lst.append(len(dfndum_set))
        tot_dfndum_lst.append(tot_dfndum_lst[-1] + len(d_lst))
        rep_ratio.append(tot_dfndum_lst[-1]/len(dfndum_set))
        try:
            art_name = art.attrib['name'].split('/')[1]
            arxiv_class = qq(art_name)
            #print(f"Found arxiv class {arxiv_class}")
        except StopIteration:
            art_name = art_name.replace('.', '/')
            try:
                arxiv_class = qq(art_name)
            except StopIteration:
                stopiterations_lst.append(art.attrib['name'])
        for D in d_lst:
            term_dict_cnt[D].update([arxiv_class])

art_name = art.attrib['name'].split('/')[1]
art_name.replace('.', '/')

len(stopiterations_lst),stopiterations_lst[:10]

# The 15 most common words are
TT = term_cnt.most_common()[:15]
for t, c  in TT:
    print(f"{t} & {c} \\\\")


# +
def baseline_dist(database=database):
    '''
    creates a Counter objects where the arXiv subject categories 
    map to the count of articles with that tag
    '''
    eng = sa.create_engine(database, echo=False)
    subject_lst = []
    with eng.connect() as con:
        #con.execute('pragma case_sensitive_like=OFF;')
        q_str = '''select tags from articles'''
        for r in tqdm(con.execute(q_str)):
            res_dict = eval(r[0])[0]
            #print(type(res_dict))
            subject_lst.append(res_dict['term'])
            
    #resu format: ("[{'term': 'math.RT', 'scheme': 'http://arxiv.org/schemas/atom', 'label': None}]",)
    return Counter(subject_lst)

bs_dist = baseline_dist()
print(bs_dist.most_common())
s = 1
Term = term_cnt.most_common()[s][0]
print(f'The term: {Term} appears in articles tagged:')
term_dict_cnt[Term]
# -

term_dict_cnt['banach space'].most_common()

# +
## Comparison of the term dist and baseline
bs_tot = sum([t[1] for t in bs_dist.items()])
term = 'banach space'
term_dist = term_dict_cnt[term].most_common()[:10]
labels, heights = list(zip(*term_dist))
term_tot = sum([t[1] for t in term_dist])
heights = [h/term_tot for h in heights]

bs_heights = [bs_dist[s]/bs_tot for s in labels]
width = 0.4
ind = np.arange(len(labels))
plt.figure(figsize=[6,4])
ax = plt.subplot(111)
plt.bar(ind, heights, width, label=term)
plt.bar(ind + width, bs_heights, width, label='baseline')
plt.xticks(ind+width/2, labels, rotation=45)
plt.legend()
#ax.get_yaxis().set_major_formatter(
#    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
#plot.title('Relative appearance of definitions')
plt.savefig('/home/luis/acl_pics/barcomp.png', dpi=160, bbox_inches='tight')
plt.show()


# +
def KLdiv(P,Q):
    if P == 0:
        return 0
    else:
        return P*np.log(P/Q)

def term_bias(term, term_dict=term_dict_cnt, bs_dist=bs_dist):
    tot_art = float(sum(bs_dist.values())) #total number of articles
    #bs_dict = {k: v/tot for k,v in bs_dist.items()}
    this_term_dict = term_dict[term]
    tot_term = float(sum(this_term_dict.values()))
    resu_dict = {}
    for k,v in bs_dist.items():
        try:
            #resu_dict[k] = this_term_dict[k]/tot_term * np.log(tot_art/bs_dist[k])
            resu_dict[k] = KLdiv(this_term_dict[k]/tot_term, bs_dist[k]/tot_art)
        except KeyError:
            resu_dict[k] = 0.0
        except ZeroDivisionError:
            print('The term: {} gave problems'.format(term))
    return resu_dict

entropy = lambda t: sum(term_bias(t).values())

#b_ = term_bias('finsler metric')
#print(sorted(b_.items(), key=lambda x: -x[1])[:10])
#sum(b_.values())

def common_low_entropy_terms(N1, N2):
    '''
    N1  # number of the most common term in glossary
    N2  # number with the least entropy
    '''
    term_entropy = []
    for t,c in term_cnt.most_common()[:N1]:
        tb = term_bias(t)
        max_subject = max(tb.items(), key=lambda x: x[1])[0]
        term_entropy.append((t, max_subject, sum(tb.values())))
    return sorted(term_entropy, key=lambda x: x[2])[:N2]


# + magic_args="echo use open_w2v instead" language="script"
# # Decode word2vec .bin file
# with open('/media/hd1/embeddings/model14-14_12-08/vectors.bin', 'rb') as mfobj:
#     m = mfobj.read()
#     #print(m[0].decode('utf8'))
#     #s = st.Struct('ii')
#     #m_it = m.__iter__()
#     head_dims = st.unpack('<11s', m[:11])
#     n_vocab, n_dim = map(int,head_dims[0].strip().split())
#     print(f"Vocabulary size: {n_vocab} and dimension of embed: {n_dim}")
#     embed = {}
#     #[next(m_it) for _ in range(11)]
#     cnt = 11
#     for line_cnt in tqdm(range(n_vocab)):
#         word = ''
#         while True:
#             next_char = st.unpack('<1s', m[cnt:cnt+1])[0].decode('utf8')
#             cnt += 1
#             if next_char == ' ':
#                 break
#             else:
#                 word += next_char
#         #print(word)
#         vec = np.zeros(n_dim)
#         for k in range(n_dim):
#             vec[k] = st.unpack('<f', m[cnt:cnt+4])[0]
#             cnt += 4
#         assert st.unpack('<1s', m[cnt:cnt+1])[0] == b'\n'
#         cnt +=1
#         embed[word] = vec
# -

term_dict_cnt['green symbol']


def ffff(x):
    return x+1
ffff.__name__

with open_w2v('/media/hd1/embeddings/model14-51_20-08/vectors.bin') as embed:
    unit_embed = {w: v/np.linalg.norm(v) for w,v in embed.items()}

common_term = term_cnt.most_common()[200][0].lower().replace(' ', '_')
print(f" The term is: {common_term} and the first components of the vector are:")
embed.get(common_term, None)[:10]

# Create a dict of "very AG" vectors
veryAG = {}
veryDG = {}
veryDict = {}
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
                veryDict[Term] = embed_vec
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
                veryDict[Term] = embed_vec
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
    for word,dist in nearest(center, unit_embed, n_near=7):
        print(word, "{0:3.2f}".format(dist))

umap1 = umap.UMAP()
tot_vec = np.stack(ag_lst + dg_lst, axis=0)
means = kmeans(tot_vec, 3)
tot_vec = np.concatenate([tot_vec, means[0]], axis=0)
labels_vec = len(ag_lst)*['math.AG'] + len(dg_lst)*['math.DG'] + len(means[0])*['center']
tran_vec = umap1.fit_transform(tot_vec, labels_vec)
x,y =  list(zip(tran_vec.transpose()))
plt.figure(figsize=[7,7])
plt.scatter(x[0][:500],y[0][:500], s=5)
plt.scatter(x[0][500:1000], y[0][500:1000], color='green', s=5)
plt.scatter(x[0][1000:], y[0][1000:], color='red' )
plt.show()

len(cc)

# +
tsne1 = TSNE()
umap1 = umap.UMAP()
plt.rcParams["image.cmap"] = 'Set1'
vec_lst = []
labels_vec = []
term_lst = []
embed_coverage_cnt = 0
#clSt = common_low_entropy_terms(100000, 50000)
for t,s,e in clSt:
    if (v := unit_embed.get(t.replace(' ', '_'))) is not None:
        embed_coverage_cnt += 1
        #if s in ['math.PR','math.AG'  ]:
        #if s in ['math.SG','math.DG'  ]:
        if s in ['math.FA','math.DG' , 'math.OC', 'math.NT' ]:
        #if s in ['math.NA','math.OC'  ]:
        #if s in ['math.PR','math.AG' , 'math.ST' ]:
        #if s in ['math.PR','math.NA' , 'math.ST' ]:
        #if s in ['math.AG','math.AT' , 'math.DG' ]:
            vec_lst.append(v)
            term_lst.append(t)
            labels_vec.append(s)
print('Embed coverage: {}%'.format(embed_coverage_cnt/len(clSt)))
pick_text = set(np.random.randint(0,len(vec_lst), size=15))
labels_set_list = list(set(labels_vec)) # to find the colormap
cc = [labels_set_list.index(l) for l in labels_vec]
tot_vec = np.stack(vec_lst, axis=0)
tran_vec = tsne1.fit_transform(tot_vec, labels_vec)
#tran_vec = umap1.fit_transform(tot_vec, labels_vec)
x,y =  list(zip(tran_vec.transpose()))
plt.figure(figsize=[10,10])
scatter = plt.scatter(x[0],y[0],marker='o', c = cc, s=20, alpha=0.8)
leg1 = plt.legend(scatter.legend_elements()[0], labels_set_list, prop={'size': 10})
for i in pick_text:
    plt.text(x[0][i], y[0][i], term_lst[i], size='x-large')
    plt.scatter([x[0][i]], [y[0][i]], s=30, c='black')
    
plt.savefig('/home/luis/acl_pics/scatter_option.png', bbox_inches='tight')
plt.savefig('/home/luis/acl_pics/scatter_option2.png', dpi=300, bbox_inches='tight')
plt.show()

# -


html = st.produce_projection_explorer(None,
                                      word2vec_model=veryDict,
                                      projection_model=umap1,
                                      category='mathAG',
                                      category_name='mathAG',
                                      not_category_name='mathDG',
                                      metadata=None)      


# + magic_args="echo skipping" language="script"
# ---- Don't run, this should be in the embed_utils.py module now
# cos_dist = lambda x, y: np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y)
# unit_embed = {w: v/np.linalg.norm(v) for w,v in embed.items()}
#
# def nearest(word_vec, n_near=10):
#     dist_dict = {}
#     unit_word_vec = word_vec/np.linalg.norm(word_vec)
#     for w, v in unit_embed.items():
#         #dist_dict[w] = cos_dist(v, word_vec)
#         dist_dict[w] = unit_word_vec.dot(v)
#     return sorted(dist_dict.items(), key=lambda pair: pair[1], reverse=True)[:n_near]

# +
# Create a "very topical" set of terms
#topic,cap = ('math.GN',3) # General Topology
#topic,cap = ('math.GT', 15) 
#topic,cap = ('math.AT', 10) #poor results
topic,cap = ('math.DG', 10) 
#topic,cap = ('math.LO', 5) 
#topic,cap = ('math.DS', 15) 
#topic,cap = ('math.PR', 15) # very "graphy" center
#topic,cap = ('math.NT', 15) 
#topic,cap = ('math.FA', 15) 
#topic,cap = ('math.GM', 2) 
#topic,cap = ('math.OC', 5) 

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

# plots the R-squared of the k-means clustering value
n_average = 5 # Number of samples to average out
dist_lst = []
for n_centers in tqdm(range(2,4)):
    mean_dist = 0
    for _ in range(n_average):
        mean_dist += kmeans(tot_vec, n_centers)[1]
    dist_lst.append(mean_dist/n_average)
plt.plot(dist_lst)
plt.show()


# +
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0, 16.0), 
                  title=None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'_inline_math_', '_display_math_', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                           'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    plt.savefig('article_wm.png')
    
#plot_wordcloud(df[df.target==1]['value vector'], title='Similar words')


# -

for name, fobj in  peep.tar_iter('/media/hd1/promath/math17/1703_004.tar.gz', '.xml'):
    if '1703.01352' in name:
        article = px.DefinitionsXML(fobj)
        art_str = " ".join([article.recutext(a) for a in article.para_list()])
plot_wordcloud(art_str, title='arXiv:1703.01352')

# Create the ripser object
rips = Rips(maxdim=2)
data = np.asarray(ag_lst)
diagrams = rips.fit_transform(data)
rips.plot(diagrams)
