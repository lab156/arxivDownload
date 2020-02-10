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

# using a list of files in the arxxmllib dataset of processed articles
# find . -name '1703.013*' > ~/find_start_1703.013.txt
with open('../find_start_1703.013.txt', 'r') as find_file:
    art_lst = find_file.readlines()

art_lst


