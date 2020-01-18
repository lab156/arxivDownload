# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
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

import extract as X
import parsing_xml as px

planetmath = etree.parse('data/planetmath_defs.xml').getroot()

mathag = etree.parse('data/mathAG_2015.xml').getroot()

len(planetmath.getchildren())

# +
ag = planetmath
G = nx.DiGraph()
def_dict = {} #keys: dfndum Value: list of hashes of statements where the is appears
hash_dict = {} # keys: hashe of statements, Values: the text of the statement
for D in ag.iter(tag = 'stmnt'):
    hash_dict[hash(D.text)] = D.text
    
for d in ag.iter(tag = 'dfndum'):
    D = d.getparent().find('stmnt').text
    if d.text.strip() in def_dict:
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

len(dgraph.nodes())

opts = {
    'node_size': 10,
    'width': 0.1,
    'with_labels': False,
    'arrow_size': 2,
}
plt.figure(1, figsize=(15,10))
pos = graphviz_layout(dgraph, prog='dot')
nx.draw_networkx(dgraph, pos, **opts)
#plt.savefig('data/starts_with_p_spectral.png')

nx.find_cycle(dgraph)

nx.drawing.nx_agraph.write_dot(dgraph, 'data/dgraph.dot')

# +
# nx.draw_networkx?
# -


