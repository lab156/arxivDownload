# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from lxml import etree 
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import os
import re
import sys
# %load_ext autoreload
# %autoreload 2
import parsing_xml as px
import random_sampling as rs

xml = etree.parse('/home/luis/media_home/xml_file.xml') 
art_dict = {}
for k, x in enumerate(xml.iter('article')):
    if x.get('searched') == 'True':
        art_dict[x.find('id').text] = x.find('location').text

art_dict['http://arxiv.org/abs/1701.00204v1']


def resolve(article_str):
    res = xml.xpath('.//article[@searched="True"]/id[contains(text(), "%s")]/ancestor::article'%article_str)
    if res:
        out = res[0].find('location').text
    else:
        out = None
    return out


ifs = 'hola'

p1 = resolve(1801.00261)
re.sub('^/mnt', '', p1)

eng = sa.create_engine('sqlite:///../arxiv1.db')
eng.connect()
SMaker = sa.orm.sessionmaker(bind=eng)
sess = SMaker()

html_p = px.DefinitionsXML('tests/latexmled_files/1501.06563_shortened.html')
html_p.get_def_sample_text_with()

import random
random.sample?

# +
qq = sess.execute('''SELECT id FROM articles 
where tags LIKE  '[{''term'': ''math.AG''%' and  
updated_parsed BETWEEN date('2017-05-01')  and date('2017-11-20') limit 20;''')

loc_path = '/home/luis/media_home/'

for l in qq:
    nonlocal_path = art_dict.get(l[0])
    if nonlocal_path:
        prepath = re.sub('^/mnt/', '', nonlocal_path)
        local_path = os.path.join(loc_path, prepath)
        xml = px.DefinitionsXML(local_path)
        tdict = xml.get_def_sample_text_with()
        print(len(tdict['real']), len(tdict['nondef']))
# -

#with open('../xml_file.xml', 'w+') as xml_file:
    #print(etree.tounicode(root, pretty_print=True), file=xml_file)

[t for t in root.getiterator(tag='article')]

[(t.attrib['processed'], t.find('name').text) for t in root.getiterator(tag='article')]

root[0].attrib['processed'] = 'True'

with open('data/file.xml', 'w+') as xml_file:
    print(etree.tostring(root, pretty_print=True).decode('utf8'),file=xml_file)

print(root[1].tag)

#T = etree.parse('../xml_file.xml')
[t[1].attrib for t in etree.iterparse('../xml_file.xml', tag='article')]
