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

rein = etree.parse('data/rein_debugged.xml').findall('par')
for k in rein:
    k.tail

rein = etree.parse('data/rein_buggy.xml')

rein1 = etree.parse('data/reinhardt/reinout.xml').findall('par')

rein1[0]


