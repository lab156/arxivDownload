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

from lxml import etree
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import pygraphviz as pgv
import random
import numpy as np

import extract as X
import parsing_xml as px

from gensim.models.callbacks import 

#planetmath = etree.parse('/media/hd1/planetmath/datasets/planetmath_definitions.xml.gz').getroot()
planetmath = etree.parse('/media/hd1/planetmath/datasets/number_theory_defs.xml').getroot()

mathag = etree.parse('data/mathAG_2015.xml').getroot()

len(planetmath.getchildren())

# +
ag = planetmath # ???
def_dict = {} #keys: dfndum Value: list of hashes of statements where the is appears
# hash_dict = {} # keys: hashes of statements, Values: the text of the statement
# for D in ag.iter(tag = 'stmnt'):
#     hash_dict[hash(D.text)] = D.text
    
for d in ag.iter(tag = 'dfndum'):
    D = d.getparent().find('stmnt').text
    if d.text.strip() in def_dict: # in case there are repeats
        def_dict[d.text.strip()].append(hash(D))
    else:
        def_dict[d.text.strip()] = [hash(D),]
print('def_dict has this many elements ', len(def_dict.values()))
# -

dgraph = nx.DiGraph()


empty_str_if_none = lambda s: s if s  else ''
for k,d_raw in enumerate(def_dict.keys()):
    d = d_raw.strip()
    if k%100 == 0:
        print('doing k=', k)
    for Def in ag.iter(tag = 'definition'):
        D = Def.find('.//stmnt')
        #Check if D is not a definition for d
        if hash(D.text) in def_dict[d]:
            pass
        else:
            dfndum_lst = [c.text for c in D.getparent().findall('.//dfndum') ]
            if d in empty_str_if_none(D.text):
                add_edges_lst = [(d, p.strip()) for p in dfndum_lst if d != p]
                dgraph.add_edges_from(add_edges_lst)

list(dgraph.nodes())[:10]

nx.shortest_path_length(dgraph,'integer', 'digamma')


# +
def bound_dist(n1, n2):
    try:
        dist = nx.shortest_path_length(dgraph, n1, n2)
    except nx.NodeNotFound:
        dist = 5
    except nx.NetworkXNoPath:
        dist = 5
    return dist

bound_dist('integer', 'amenable')
# -

node_lst = list(dgraph.nodes())
delta_list = []
for k in range(1000):
    x,y,z,t = random.sample(node_lst, 4)
    s1 = bound_dist(x,y) + bound_dist(z,t)
    s2 = bound_dist(x,z) + bound_dist(y,t)
    s3 = bound_dist(x,t) + bound_dist(z,y)
    big1, big2 = sorted([s1,s2,s3], reverse=True)[:2]
    delta = 0.5*(big1 - big2)
    delta_list.append(delta)
print('Approx delta-hyperbolicity: {}'.format(sum(delta_list)/float(len(delta_list))))

opts = {
    'node_size': 10,
    'width': 0.1,
    'with_labels': False,
    'arrowsize': 2,
}
plt.figure(1, figsize=(15,10))
pos = graphviz_layout(dgraph, prog='dot')
nx.draw_networkx(dgraph, pos, **opts)
plt.savefig('data/number_theory_dgraph.png')

nx.find_cycle(dgraph)

# + magic_args="echo This save the image to a file" language="script"
# nx.drawing.nx_agraph.write_dot(dgraph, 'data/dgraph.dot')

# +
# nx.draw_networkx?
