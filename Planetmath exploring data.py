# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
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

# %load_ext autoreload
# %autoreload 2
import parsing_xml as px

# +
Types = coll.Counter()
def_files = []
tex_filepaths = glob.glob('/mnt/planetmath/*/*.tex')
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
            def_files.append(f)
        
        
    
# -

Types

def_files[:10]

pmxml = px.PlanetMathXML('../11_Number_theory/11A05-SphenicNumber.xml')
para_lst = pmxml.para_list()
print(etree.tostring(pmxml.create_xml_branch(), pretty_print=True).decode('utf8'))
#pmxml.extract_text()
pmxml.tag_vals

ns = {'latexml': 'http://dlmf.nist.gov/LaTeXML' }
pmfile = etree.parse('../11_Number_theory/11E39-SymmetricBilinearForm.xml')
tag_lst = pmfile.xpath('.//latexml:ERROR', namespaces=ns)
for t in tag_lst:
    print(t.text, t.tail)

# +
# Create a root to store all definitions
root = etree.Element('root')

for filenm in glob.glob('../11_Number_theory/*.xml'):
    try:
        px_file = px.PlanetMathXML(filenm)
        if px_file.tag_vals['pmtype'] and px_file.tag_vals['pmtype'][0] == 'Definition':
            branch = px_file.create_xml_branch()
            root.append(branch)
    except ValueError as e:
        print('%s is empty!'%filenm)
    
#print(etree.tostring(root, pretty_print=True).decode('utf8'))
# -

with open('../planetmath_defs.xml', 'w+') as stack_file:
    stack_file.write(etree.tostring(root, pretty_print=True).decode('utf8'))


