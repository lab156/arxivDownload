# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import datetime as dt
import glob
import pickle
import re
import dateutil
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import collections as coll
import pandas as pd
import tarfile
from functools import reduce
import magic
from collections import defaultdict
import itertools
from functools import reduce
# %matplotlib inline  

# %load_ext autoreload
# %autoreload 2
import latexml_err_mess_stats as Err

with open('../stats.pickle', 'rb') as fh:
    stat_lst = pickle.load(fh)

# +
#Grand totals
# SUCC    # everything went fine as far as I can tell
# TIMED   # timed out -- used more than the allowed time 1200 or 2400
# FATAL   # at least one fatal error found
# MAXED   # maxed out the allowed number of errors 100,000
# DIED    # found dead: ex. no finished processing timestamp (mostly out of memory errors)
# NOTEX   # No TeX file was found it might be pdf only or a weird case like 1806.03429
# #NOLOG  # No log file, different from DIED and NOTEX
# FAIL = FATAL | TIMED | MAXED | DIED

pvec = sum(np.array(l[1][2]) for l in stat_lst)
tot = float(pvec[0] + pvec[1])

str_fmt = lambda st: "{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(*list(st))
title = ["Success", "Fail", "Fatal", "Maxed", "Timed", "Died", "no_tex"]
print(str_fmt(title))
print(str_fmt(pvec))
print(str_fmt(np.round(pvec*100/tot, decimals=1)))
# -

# ## Size of the Processed file
# * 7.0K    math91
# * 8.9M    math92
# * 8.5M    math93
# * 13M     math94
# * 14M     math95
# * 15M     math96
# * 22M     math97
# * 132M    math98
# * 169M    math99
# * 212M    math00
# * 233M    math01
# * 306M    math02
# * 371M    math03
# * 481M    math04
# * 574M    math05
# * 818M    math06
# * 941M    math07
# * 901M    math08
# * 25G     math09
# * 24G     math10
# * 29G     math11
# * 24G     math12
# * 1.8G    math13
# * 2.0G    math14
# * 2.2G    math15
# * 2.4G    math16
# * 2.5G    math17
# * 2.7G    math18
# * 2.8G    math19
# * 975M    math20
# * 124G    *total*

# + jupyter={"outputs_hidden": true}
stat_lst = []
mem_err_lst = []
common_path = '/mnt/promath/math05/'
for walk in os.walk('/mnt/promath'):
    for fname in walk[2]:
        if '.tar' in fname:
            try:
                st = Err.open_tar(os.path.join(walk[0], fname))
                print(fname, '  ', st[2])
                stat_lst.append((fname,st))
            except:
                ee = sys.exc_info()
                mem_err_lst.append((fname, ee))

# + jupyter={"outputs_hidden": true}
Err.open_tar('/home/pi/0508_002.tar.gz')[2]
# -

mem_err_lst

try:
    raise ReadError()
except:
    ee = sys.exc_info()
    print('lived', ee)

r = '/mnt/promath/math05/0501_001/math.0501207'
Err.open_dir(r)

list(map(lambda x:x.decode(), pp.commentary))

article_dictfiles = glob.glob('data/problem_files_starting_1703/*/latexml_erro*')
err.summary(lst_error_files)

p_lst = list(map(err.ParseLaTeXMLLog, lst_error_files))
p_times = [p.time_secs for p in p_lst]
Cut,bins = pd.cut(p_times, 8, retbins=True)
count = coll.Counter(Cut)
for c in sorted(list(count)):
    print(c, count[c])

encoding_lst = []
for l in lst_error_files:
    p = err.ParseLaTeXMLLog(l)
    if p.fatal_errors:
        print('fe: %s ee: %s fn: %s'%(p.fatal_errors, p.errors, p.filename))
        print(p.commentary()[-1])
        G = re.search('Finished in less than (\d+) seconds', p.commentary()[-1]).group(1)
        
        print(G)
        print('-----------------')

I = lst_error_files.index(list(filter(lambda s: '1703.01352' in s, lst_error_files ))[0])
with open(lst_error_files[20], 'r') as err_file:
    err = err_file.read()
print(re.search('\nConversion complete:(.*)', err).group(0))    
print(re.search('\nConversion complete: (No obvious problems\.)?(\d+ warnings?[;\.] ?)?(\d+ errors?[;\.] ?)?(\d+ fatal errors?[;\.] ?)?(\d+ undefined macros?\[[\*\@\{\}\\\\,\w\. ]+\][;\.] ?)?(\d+ missing files?\[[,\w\. ]+\])?.*\n', err).groups())


f = filter(lambda x: 'errors' in x, val)
len(f)


# +
def time_from_latexml_log(f):
    '''
   return time in seconds that a latexml process spent as it appears on the logs
   time is returned in seconds
    '''
    with open(f,'r') as open_file:
        file_content = open_file.read()
        start = re.search('\nprocessing started (.*)\n', file_content).group(1)
        finish = re.search('\nprocessing finished (.*)\n', file_content).group(1)
        d1 = dateutil.parser.parse(start)
        d2 = dateutil.parser.parse(finish)
    return (d2-d1).seconds

time_from_latexml_log('data/nanopterons2/latexml_errors_mess.txt')/60.0
# -

duration_lst = []
for f in lst_error_files:
    with open(f,'r') as open_file:
        file_content = open_file.read()
        start = re.search('\nprocessing started (.*)\n', file_content).group(1)
        finish = re.search('\nprocessing finished (.*)\n', file_content).group(1)
        d1 = dateutil.parser.parse(start)
        d2 = dateutil.parser.parse(finish)
        duration_lst.append((d2-d1).seconds/60.0)

plt.hist(duration_lst, bins=30)


