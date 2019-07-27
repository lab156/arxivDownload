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

import pyparsing as pp

#ssn ::= num+ '-' num+ '-' num+
#num ::= '0' | '1' | '2' etc
dash = '-'
ssn = pp.Combine(pp.Word(pp.nums, exact=3) +
                 dash + pp.Word(pp.nums, exact=2) +
                 pp.Suppress('-') + pp.Word(pp.nums, exact=4))
target = '123-45-6789'
result = ssn.parseString(target)
print(result)

with open('../tests/tex_files/reinhardt/reinhardt-optimal-control.tex', 'r') as rein_file:
    rein = rein_file.read()

bs = '\\'
tikzfig = Word(bs + 'tikzfig')
tikzfig.searchString(rein)


