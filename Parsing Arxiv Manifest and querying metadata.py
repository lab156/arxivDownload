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

import xml.etree.ElementTree as ET
import pandas as pd
with open('../manifest_jan_2019.xml', 'r') as f:
    mani = ET.parse(f)

# +
root = mani.getroot()
def parse_element(elem):
    return_dict = {}
    for e in elem:
        return_dict[e.tag] = e.text
    return return_dict
def parse_root(root):
        return [parse_element(child) for child in iter(root) if child.tag != 'timestamp']

filedf = pd.DataFrame(parse_root(root))
filedf[['num_items', 'size']] = filedf[['num_items', 'size']].astype(int)
filedf['filename'] = filedf['filename'].astype(str)
# -

import hashlib 

hash_md5 = hashlib.md5()
with open('../arXiv_src_1804_001.tar', "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
hash_md5.hexdigest()

s = filedf.iloc[0]; s

len(s['filename'])

import subprocess
P = subprocess.run(['/bin/s3cmd', 'get', '--requester-pays', 's3://arxiv/src/arXiv_src_1804_006.tar', '..'], 
                   stderr=subprocess.PIPE,
                   stdout=subprocess.PIPE)

P.stdout

P.stderr

P.check_returncode()

df.filename.isin(notdf.filename)

df.to_csv('head_df.csv')

notdf.to_csv('notdf.csv')

# + jupyter={"outputs_hidden": true}
nondf = pd.read_csv('/home/luis/mountpoint/arXiv_src/downloaded_log.csv',index_col=0)
nondf.append(pd.DataFrame([{'filename':'hola'}]), ignore_index=True)
# -

import os
os.path.ismount('/mnt')

notdf

# %load_ext autoreload
# %autoreload 2

from dload import DownloadMan, parse_manifest
mountpoint = '/mnt/arXiv_src/'
allfiles = 'allfiles2.csv'
doun = 'downloaded_log.csv'
error_log = 'error.log'
D = DownloadMan(mountpoint, allfiles, doun, error_log)

import process as P
import re
name = re.match(r'.*([a-z]+)([0-9]{7}).+', '0703/math0703071.gz') 
name.group(1)
P.tar2api2('0703/math0703071.gz', sep='*')
P.detect_format('0701/1701.00112')

# %time xx.extract_tar('../check4', 'math.DG')

[s for s in xx.art_lst if '0701590' in s]

ll = list(range(13))
length = 4
lal = []
i = 0
while i*length < len(ll):
    seclen = min((i+1)*length, len(ll))
    lal += list(range(i*length, seclen))
    i += 1
print(lal)

xx.path_dir('../mathy')

D.check_md5()

new_all = parse_manifest('../arXiv_src_manifest_Oct_2019.xml')

new_all.to_csv('../arXiv_src_manifest_Oct_2019.csv')

# + jupyter={"outputs_hidden": true}
for el in res.getchildren():
    print(el.tag)
# -

import sys
sys.path.insert(0,'arxiv.py/')
import arxiv

import arxiv
D = arxiv.query(id_list=['1601.00690'])
len(D)
#for d in D[0].keys():
#    print(d,D[0][d])

# + jupyter={"outputs_hidden": true}
for k in D[0].keys():
    print(k , ' :: ' , D[0][k])
# -

len(repr(D[0]['tags']))

# + jupyter={"outputs_hidden": true}
# arxiv API usage
D = arxiv.query(id_list=['1601.00105'])
for k in D[0].keys():
    print(k,D[0][k])
# -

arxiv.query(id_list=['1805.02773'])[0].links

year_month = '1804.'
id_list = [year_month + str(k).zfill(5) for k in range(1,263)]
dicts = arxiv.query(id_list=id_list, max_results=300)

# + jupyter={"outputs_hidden": true}
diff_geom = [d for d in dicts if d.get('tags',None)[0]['term']=='math.DG']
diff_geom
# -

[d.title for d in diff_geom]

math = [d for d in dicts if 'math' in d['tags'][0]['term']]

# + jupyter={"outputs_hidden": true}
set([d['tags'][0]['term'] for d in math])

# +
import tarfile

with tarfile.open('tests/minitest.tar') as ff:
    art_lst = [k for k in ff.getmembers()]
print(art_lst)

# +
import magic
with tarfile.open('tests/minitest2.tar') as ff:
    for f in ff:
        print(f)
        
ff.close()
# -

ff = tarfile.open('tests/minitest.tar')
fi = ff.getmembers()[2]
fobj = ff.extractfile(fi.name)
print("1)  ", magic.detect_from_content(fobj.read(2048)))
fobj.seek(0)
unz_file = tarfile.open(fileobj=fobj, mode='r:gz')
unz_file.getmembers()
#unz_file = gzip.open(fobj, 'rb')
#print("2)  ", magic.detect_from_content(unz_file.read(2048)))
#unz_file.seek(0)
#unz_tar = tarfile.open(fileobj=unz_file)
#print(unz_tar.getmembers())

# + jupyter={"outputs_hidden": true}
with tarfile.open('tests/minitest2.tar') as ff:
    for fi in ff.getmembers()[1:]:
        fobj = ff.extractfile(fi.name)
        the_magic = magic.detect_from_content(fobj.read(2048))
        fobj.seek(0)
        print(fi.name,the_magic.name)
        if 'gzip compressed' in the_magic.name:
            try:
                with tarfile.open(fileobj=fobj, mode='r:gz') as unzipped_file:
                    #snd_magic = magic.detect_from_fobj(unzipped_file)
                    #unz_tarinfo = unzipped_file.next()
                    pass
                    #if unz_tarinfo:
                    #    print("     * is regular ", unz_tarinfo.isreg())
                    #unzipped_file.seek(0)
                    #if snd_magic.mime_type == 'application/x-tar':
                    #    unzipped.getmembers()
            except gzip.BadGzipFile:
                print('gave me badgzipfile')
# -

pro.

import gzip
with tarfile.open('tests/minitest2.tar') as ff:
    for fi in ff.getmembers()[1:]:
        fobj = ff.extractfile(fi.name)
        the_magic = magic.detect_from_content(fobj.read(2048))
        fobj.seek(0)
        print(pro.Tarfi.name,the_magic.name)
        if '.tex.cry"' in the_magic.name:
            print("cry baby cry")
        else:
            try:
                with gzip.open(fobj,'rb') as unzipped_file:
                    snd_magic = magic.detect_from_content(unzipped_file.read(2048))
                    unzipped_file.seek(0)
                    print("     *", snd_magic.name)
                    if snd_magic.mime_type == 'application/x-tar':
                        with tarfile.open(fileobj=unzipped_file) as tars:
                            print("     * There are ", len(tars.getmembers()), 'items')
            except gzip.BadGzipFile:
                print('gave me badgzipfile')
        print(' ')

print(snd_magic.mime_type)
print(snd_magic.encoding)
print(snd_magic.name)

with open('./zhu_untared.tex', 'w') as f:
    f.write(zhu_str.decode('utf-8'))

# + jupyter={"outputs_hidden": true}
print(zhu_str)

# +
import gzip
import chardet
import tarfile
file_tar = tarfile.open('/mnt/arXiv_src/src/arXiv_src_9904_001.tar')
#for f in file_tar.getmembers():
#    print(f.get_info())

tar2 = file_tar.extractfile('9904/math9904086.gz')
#print(chardet.detect(tar2))
#with open('../todo_macizo.pdf','w') as todo_macizo:
#    todo_macizo.write(tar2.read().decode('utf-8'))
#with tarfile.open(fileobj=tar2) as tars:
#    for t in tars:
 #       print(t.get_info()['name'])

#gz = gzip.open(tar2, 'rb')

with gzip.open(tar2,'r') as tars:
    t = tars.read()


#print(t.decode('utf-8'))
chardet.detect(t)

#with open('../holita.txt','w') as holaf:
#    holaf.write(allfile.read().decode('utf-8'))
    

# + jupyter={"outputs_hidden": true}
t.decode('cp932')
#t.decode(errors='ignore')
# -

try:
    t.decode()
except UnicodeDecodeError as e:
    print('todo bien')
    print(e)
    

import difflib
Diffs = difflib.unified_diff(t.decode(errors='ignore').splitlines(), t.decode('koi8-r')) 
for d in Diffs:
    print(d)

tarfile.ReadError()

with tarfile.open('tests/minitest.tar') as morgan:
    for mor in morgan:
        print(mor.name)


# +
# file_tar.extract?

# +
# tarfile.TarFile.extractall?
# -

import chardet
chardet.detect(t)

import glob
glob.glob('/mnt/arXiv_src/src/arXiv_src_1805_*')

# +
# bytes.decode?
# -

type(t)

import shutil
shutil.rmtree('check_test/')

with open('../check2/1601.00103/1601.00103.tex','r') as fu:
    print(fu.readlines()[:10])

# + jupyter={"outputs_hidden": true}
os.listdir('../check6')
# -

set( ['commentary.txt', 'definitions.sty', 'hyperplane.bbl', 'hyperplane.tex'])== set(['commentary.txt', 'hyperplane.tex', 'definitions.sty', 'hyperplane.bbl'])


