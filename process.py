import sys
sys.path.insert(0,'arxiv.py/')
import arxiv
import tarfile
import gzip
import re
import os.path

def api2tar(id_str):
    '''
    The arxiv API produces ids in the format:
    http://arxiv.org/abs/1804.00018v1',
 http://arxiv.org/abs/1804.00193v1',
 http://arxiv.org/abs/1804.00259v2
    The tar files have name format:
    <TarInfo '1804/1804.00020.gz' at 0x7f0ba68c8cc8>
    '''
    name = re.match(r'.*([0-9]{4})\.([0-9]{5})',id_str)
    return name.group(1) + '/' + name.group(1) + '.' + name.group(2) + '.gz'

def tar2api(id_str):
    '''
    The name of the files when untared has the format:
    '1804/1804.00020.gz', '1804/1804.00239.gz', '1804/1804.00127.gz', '1804/1804.00197.gz'
    in order to get a searcheable id in the API take the leading number and the file type
    '''
    name = re.match(r'.*([0-9]{4})\.([0-9]{5})',id_str)
    return name.group(1) + '.' + name.group(2) 
    
class Xtraction(object):
    def __init__(self, tar_path, *argv, **kwarg):
        #get a list of all objects in the file at tar_path
        #these have name format: '1804/1804.00020.gz'
        if os.path.isfile(tar_path):
            self.tar_path = tar_path
        else:
            raise(ValueError('There is no file at %s'%tar_path))
        print('untaring file: %s'%tar_path)
        with tarfile.open(tar_path) as ff:
            self.art_lst = [k.get_info()['name'] for k in ff.getmembers()]

        #To query an id, from  '1804/1804.00020.gz' we just get 1804.00020
        query_id_list = [tar2api(s) for s in self.art_lst[1:]]

        print('querying the arxiv')
        self.query_results = arxiv.query(id_list=query_id_list,
                max_results=len(self.art_lst))
        print('query successful')

    def filter_MSC(self, MSC , run_api2tar=True):
        if run_api2tar:
            return [api2tar(d.id) for d in self.query_results \
                    if d['tags'][0]['term']==MSC]
        else:
            return [d for d in self.query_results if d['tags'][0]['term']==MSC]

    def extract2str(self, filename):
        '''
        given filename, extract a file with that name from
        self.tar
        '''
        with tarfile.open(self.tar_path) as ff:
            file_gz = ff.extractfile(filename)
            with gzip.open(file_gz,'rb') as fb:
                file_str = fb.read()
        return file_str.decode('utf-8')
    



