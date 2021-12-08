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
import os
import glob
import gzip
from lxml import etree

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from embed.clean_and_token_text import normalize_text

pars = etree.XMLParser(recover=True)
# -

NN_path = '/home/luis/NNglossary/'
SGD_path = '/media/hd1/glossary/v3/'
File_lst_NN = glob.glob(NN_path + 'math*/*')
File_lst_SGD = glob.glob(SGD_path + 'math*/*')

# Simple example to parse files 
xml_root = etree.parse(File_lst_NN[0]).getroot()
print(etree.tostring(xml_root[0], pretty_print=True).decode('utf-8'))

# #%%script echo run with function below instead
dfndum_set = set()
new_dfndum_lst = [0]
tot_dfndum_lst = [0]
rep_ratio = []
term_cnt = Counter()
norm_term_cnt = Counter()
perc_array = np.array([])
for xml_path in tqdm(glob.glob(NN_path + 'math*/*.xml.gz')):
    gtree = etree.parse(xml_path).getroot()
    for art in gtree.iter(tag='article'):
        d_lst = [d.text for d in art.findall('.//dfndum')]
        dfndum_set.update(d_lst)
        term_cnt.update(d_lst)
        norm_term_cnt.update([normalize_text(d, 'rm_punct') for d in d_lst])
        new_dfndum_lst.append(len(dfndum_set))
        tot_dfndum_lst.append(tot_dfndum_lst[-1] + len(d_lst))
        rep_ratio.append(tot_dfndum_lst[-1]/len(dfndum_set))
        
        N = float(art.attrib['num'])
        percs = np.array(list(float(a.attrib['index']) for a in art.findall('.//definition')))/N
        perc_array = np.append(perc_array, percs)
        #if len(dfndum_set)%100000 == 0:
        #    print(len(dfndum_set))


# +
def read_all_files(Path):
    dfndum_set = set()
    new_dfndum_lst = [0]
    tot_dfndum_lst = [0]
    rep_ratio = []
    term_cnt = Counter()
    norm_term_cnt = Counter()
    #perc_array = np.array([])
    for xml_path in tqdm(glob.glob(Path + 'math*/*.xml.gz')):
        gtree = etree.parse(xml_path).getroot()
        for art in gtree.iter(tag='article'):
            d_lst = [d.text for d in art.findall('.//dfndum')]
            dfndum_set.update(d_lst)
            term_cnt.update(d_lst)
            norm_term_cnt.update([normalize_text(d, 'rm_punct') for d in d_lst])
            new_dfndum_lst.append(len(dfndum_set))
            tot_dfndum_lst.append(tot_dfndum_lst[-1] + len(d_lst))
            rep_ratio.append(tot_dfndum_lst[-1]/len(dfndum_set))

            N = float(art.attrib['num'])
            #percs = np.array(list(float(a.attrib['index']) for a in art.findall('.//definition')))/N
            #perc_array = np.append(perc_array, percs)
    return norm_term_cnt, term_cnt
    
sgd_ntc, sgd_tc = read_all_files(SGD_path)
nn_ntc, nn_tc = read_all_files(NN_path)

# +
sgd_set = set(sgd_ntc.keys())
nn_set = set(nn_ntc.keys())
In= sgd_set.intersection(nn_set)
Un = sgd_set.union(nn_set)
nn_sgd_diff = nn_set.difference(sgd_set)
sgd_nn_diff = sgd_set.difference(nn_set)

In_tot_cnt = sum([sgd_ntc[t] + nn_ntc[t] for t in In])
nn_sgd_tot_cnt = sum([sgd_ntc[t] + nn_ntc[t] for t in nn_sgd_diff])
sgd_nn_tot_cnt = sum([sgd_ntc[t] + nn_ntc[t] for t in sgd_nn_diff])
print('The Intersection has {:,} -- {:1.2f}% has total cnt: {:,}'.format(len(In),
                                                                       len(In)/len(Un),
                                                                      In_tot_cnt))
print('NN - SGD has         {:,} -- {:1.2f}% has total cnt: {:,}'.format(len(nn_sgd_diff),
                                                     len(nn_sgd_diff)/len(Un),
                                                    nn_sgd_tot_cnt))

print('SGD - NN has         {:,} -- {:1.2f}% has total cnt: {:,}'.format(len(sgd_nn_diff),
                                                     len(sgd_nn_diff)/len(Un),
                                                                        sgd_nn_tot_cnt))
In_cnt = Counter({t:sgd_ntc[t] + nn_ntc[t] for t in In })
print("The most common terms in the intersection are:")
In_cnt.most_common()[:25]
# -

# number of terms in a Defdum
fig=plt.figure(figsize=(6, 5))
n_words = [len(w.split()) for w in In_cnt.keys()]
len_cnt = Counter(n_words)
plt.style.use('ggplot')
reversed_range = list(range(1,11))
reversed_range.reverse()
ax = plt.subplot(111)
plt.barh([k for k,_ in enumerate(reversed_range)], [len_cnt[l] for l in reversed_range],color='orange')
plt.yticks(range(10), reversed_range)
plt.ylabel('Number of Words in Phrase')
plt.xlabel('Count of Terms with this Length')
plt.title('Length of Definienda')
ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.show()
#[w for w in dfndum_set if len(w.split()) > 10][:5]
print("The number of terms with more than 6 words is: {}".format(
sum([1 for n in n_words if n > 6])))
print('\n'.join([w for w in In_cnt if len(w.split()) > 6]))

print('Search for a term')
next((ind,ph) for ind, ph in enumerate(In_cnt.most_common()) if ph[0] == 'well known method')
print('Search for a word')
[T for T in In_cnt.keys() if 'injectivity' in T]

print("Total # of term after normalized text: {:,d}".format(len(nn_ntc)))
print(f"# of distinct terms: {new_dfndum_lst[-1]:,d}")
s = 0
nn_ntc.most_common()[s:s+15]
phrases_cnt = [ph for ph in nn_ntc.most_common() if len(ph[0].split()) > 1]
phrases_cnt[s:s+15]
#term_cnt['local stability properties']


print(f"Total # of term: {tot_dfndum_lst[-1]:,d}")
print(f"# of distinct terms: {new_dfndum_lst[-1]:,d}")
s = 0
print(f"The most common unnormalized terms are:")
term_cnt.most_common()[s:s+15]
#term_cnt['local stability properties']


# +
#plt.plot(term_cnt.values())
plt.figure(figsize=[12,9])
ax1 = plt.subplot(221)
plt.plot([t[1] for t in term_cnt.most_common()][0:25])
ax1.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: '{:,d}'.format(int(x))))
plt.title("Most common [0,25]")
ax2 = plt.subplot(222)
plt.title("Most common [0,100]")
plt.plot([t[1] for t in term_cnt.most_common()][0:100])
ax2.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: '{:,d}'.format(int(x))))

ax3 = plt.subplot(223)
plt.plot([t[1] for t in term_cnt.most_common()][1000:])
ax3.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: '{:,d}'.format(int(x+1000))))
plt.title(">1000 Tail")
ax4 = plt.subplot(224)
tail_val = 100000
plt.plot([t[1] for t in term_cnt.most_common()][tail_val:])
ax4.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: '{}K'.format(int((tail_val+x)/1000))))
plt.title(f">{tail_val:,d} Tail")
plt.show()


# +
plt.figure(figsize=(12,6))
ax1 = plt.subplot(121)
plt.plot(new_dfndum_lst, label='new')
plt.plot(tot_dfndum_lst, label='total')
#plt.title('New terms in math.AG 2015')
plt.ylabel('Number of terms')
plt.xlabel('Number of articles')
ax1.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: '{:,d}'.format(int(x))))
ax1.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: '{:,d}K'.format(int(x/1000))))

plt.legend()
ax2 = plt.subplot(122)
plt.plot(rep_ratio)
plt.yticks(np.arange(1,8,0.5))
plt.grid(True)
ax2.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: '{:,d}K'.format(int(x/1000))))
plt.title('Ratio Total/New Terms')
plt.savefig('/home/luis/overview_pics/new_repeat_terms.png')
plt.show()

# -

# number of terms in a Defdum
fig=plt.figure(figsize=(6, 5))
n_words = [len(w.split()) for w in dfndum_set]
len_cnt = Counter(n_words)
plt.style.use('ggplot')
reversed_range = list(range(1,11))
reversed_range.reverse()
ax = plt.subplot(111)
plt.barh([k for k,_ in enumerate(reversed_range)], [len_cnt[l] for l in reversed_range],color='green')
plt.yticks(range(10), reversed_range)
plt.ylabel('Number of Words in Phrase')
plt.xlabel('Count of Terms with this Length')
plt.title('Length of Definienda')
ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.savefig('/home/luis/overview_pics/len_of_terms.png')
plt.show()
#[w for w in dfndum_set if len(w.split()) > 10][:5]
print("The number of terms with more than 6 words is: {}".format(
sum([1 for n in n_words if n > 6])))


plt.figure(figsize=[9,6])
ax = plt.subplot(111)
plt.hist([p for p in perc_array if p<=1.0],100)
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.title('Relative appearance of definitions')
plt.savefig('/home/luis/overview_pics/cum_defs.png')
plt.show()
print('Number of indices greater than article number {}'.format(
sum([1 for p in perc_array if p>=1.0])))

for xml_path in tqdm(glob.glob(NN_path + 'math*/*.xml.gz')):
    gfile = etree.parse(xml_path)
    for art in gfile.findall('.//article'):
        N = int(art.attrib['num'])
        for d in art.findall('.//definition'):
            if int(d.attrib['index']) > N:
                print("Name: {} -- index: {} -- num: {}".format(art.attrib['name'], d.attrib['index'], N))
