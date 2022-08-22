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
import glob
import re
import collections as coll
from lxml import etree
import html
import matplotlib.pyplot as plt
import random
import gzip

# %load_ext autoreload
# %autoreload 2
import parsing_xml as px

# +
Types = coll.Counter()
def_files = []
def_paths = []
tex_filepaths = glob.glob('/media/hd1/planetmath/*/*.tex')
for f in tex_filepaths:
    with open(f, 'r') as f_file:
        text = f_file.read()
        results = re.findall('\\\\pmtype\{(.*)\}', text)
        Types.update(results)
        try:
            assert(len(results)==1)
        except AssertionError:
            print("the file ", f, " has ", len(results), " results")
        if results and results[0] == 'Definition':
            def_paths.append(f)
            def_files.append()
        
        
    
# -

Types

#changes the path `s` ending from .tex to .xml
switch2xml = lambda s: s[:-4] + '.xml'
switch2xml(def_files[0])

root = etree.Element('root')
with_ddum_cnt = 0
for def_file in def_files:
    try:
        pmxml = px.PlanetMathXML(switch2xml(def_file))
        para_lst = pmxml.para_list()
        branch = pmxml.create_xml_branch()
        if branch.find('.//dfndum') is not None:
            with_ddum_cnt += 1
            root.append(branch)
    except ValueError:
        print(f'{def_file} is empty')
print(with_ddum_cnt)

all_articles = root.findall('.//article')
random_article = random.sample(all_articles, 1)[0]
print(random_article.find('.//definition')[0].text)

#print(etree.tostring(random_article, pretty_print=True).decode('utf8'))
print(random_article.find('.//dfndum').text)

# Lengths of the definitions to get the max_seq_len parameter
plt.figure(figsize=[9,6])
ax = plt.subplot(111)
plt.hist([min(len(s.text), 10000) for s in root.findall('.//stmnt')], 100)
plt.grid()
plt.title('Length in characters of the definitions in the training set')
plt.show()

# #%%script echo skip
# Write the root to a xml.gz file
with gzip.open('/media/hd1/planetmath/datasets/planetmath_definitions.xml.gz', 'w') as xfobj:
    xfobj.write(etree.tostring(root, pretty_print=True))

ns = {'latexml': 'http://dlmf.nist.gov/LaTeXML' }
pmfile = etree.parse('/media/hd1/planetmath/11_Number_theory/11E39-SymmetricBilinearForm.xml')
tag_lst = pmfile.xpath('.//latexml:ERROR', namespaces=ns)
for t in tag_lst:
    print(t.text, t.tail)

with open('/media/hd1/planetmath/11_Number_theory/11-00-Coprime.tex') as tex_fobj:
    results = re.findall('\\\\(pm[a-z]+)\{(.*?)\}', tex_fobj.read())
print(results)

px_f = px.PlanetMathXML('/media/hd1/planetmath/91_Game_theory_economics_social_and_behavioral_sciences/91A99-NashEquilibrium.xml')
px_f.extract_text()

# +
# Create a root to store all definitions
root = etree.Element('root')

for filenm in glob.glob('/media/hd1/planetmath/91_Game_theory_economics_social_and_behavioral_sciences/*.xml'):
    try:
        px_file = px.PlanetMathXML(filenm)
        if px_file.tag_vals['pmtype'] and px_file.tag_vals['pmtype'][0].lower() == 'definition':
            if px_file.tag_vals['pmtitle'][0].lower() in px_file.extract_text().lower():
                print(px_file.tag_vals['pmtitle'], px_file.file_path)
            #branch = px_file.create_xml_branch()
            #root.append(branch)
    except ValueError as e:
        print('%s is empty!'%filenm)
    
#print(etree.tostring(root, pretty_print=True).decode('utf8'))

# + magic_args="echo careful this writes to disk" language="script"
# with open('../planetmath_defs.xml', 'w+') as stack_file:
#     stack_file.write(etree.tostring(root, pretty_print=True).decode('utf8'))
# -

random_article.find('.//stmnt').text

2+2


