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

import datetime as dt
import glob
import re
import dateutil
import matplotlib.pyplot as plt
import os
import numpy as np
import collections as coll
import pandas as pd
import tarfile
import magic
from collections import defaultdict
# %matplotlib inline  

# %load_ext autoreload
# %autoreload 2
import latexml_err_mess_stats as err

magic.detect_from_filename('data/0808_001.tar')

err.summary('../0804_001.tar', print='fail')

# + jupyter={"outputs_hidden": true}
with tarfile.open('../0804_001.tar') as tar_file:
    val = list(filter(lambda x: '0804.1905' in x, tar_file.getnames()))
    comm = tar_file.extractfile(next(filter(commentary_pred, val)))
    log_name = next(filter(error_log_pred, val), None)
    if log_name:
        log = tar_file.extractfile(log_name)
    else:
        log = None
    pp = err.ParseLaTeXMLLog(log, comm, name)
print(name,pp.log)

# +
article_dict = defaultdict(list)
commentary_pred = lambda x: 'latexml_commentary' in x
error_log_pred = lambda x: 'latexml_errors' in x
with tarfile.open('../0808_003.tar') as tar_file:
    article_set = set()
    for pathname in tar_file.getnames():
        dirname = pathname.split('/')[1]
        article_set.add(dirname)
        article_dict[dirname].append(pathname)
    for name,val in article_dict.items():
        comm = tar_file.extractfile(next(filter(commentary_pred, val)))
        log_name = next(filter(error_log_pred, val), None)
        if log_name:
            log = tar_file.extractfile(log_name)
        else:
            log = None
        the_name = name
            #print(log, ' ', comm, ' ')
        pp = err.ParseLaTeXMLLog(log, comm, name)
        print(name,pp.result)
            

# -

list(map(lambda x:x.decode(), pp.commentary))

article_dictfiles = glob.glob('data/problem_files_starting_1703/*/latexml_erro*')
err.summary(lst_error_files)

p_lst = list(map(err.ParseLaTeXMLLog, lst_error_files))
p_times = [p.time_secs for p in p_lst]
Cut,bins = pd.cut(p_times, 8, retbins=True)
count = coll.Counter(Cut)
for c in sorted(list(count)):
    print(c, count[c])

type(np.NAN)

for c in Cut:
    print(c.)

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


