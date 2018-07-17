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

def tar_id(id_str):
    '''
    Given a path like: /mnt/arXiv_src/src/arXiv_src_1804_001.tar
    return the unique identifier 1804_001
    '''
    name = re.match(r'.*arXiv_src_([0-9]{4})_([0-9]{3})\.tar', id_str)
    return name.group(1) + '_' + name.group(2)

    
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

    def path_dir(self, output_dir):
        '''
        Create if not exists dir with path:
        output_dir
          |_
            1804.0001   (tar_id name)
        '''
        the_path = os.path.join(output_dir, tar_id(self.tar_path))
        if not os.path.exists(the_path):
            os.makedirs(the_path)
        return the_path


    def extract_any(self, filename, output_dir):
        '''
        given filename, decide it is a tape archive (tar)
        or gzipped compressed file and extract it to the directory output_dir
        self.tar
        Example
        filename: /mnt/arXiv_src/src/arXiv_src_1804_001.tar
        output_dir: math.DG
        '''
        short_name = tar2api(filename) # format 
        output_path = os.path.join(self.path_dir(output_dir), short_name)
        os.mkdir(output_path)
        with tarfile.open(self.tar_path) as ff:
            file_gz = ff.extractfile(filename)
            #if tarfile.is_tarfile(file_gz):
#                with tarfile.open(file_gz) as ftar: 
#                    ftar.extractall(output_path)
#            else:
            with gzip.open(file_gz,'rb') as fgz:
                file_str = fgz.read()
            try:
                decoded_str = file_str.decode('utf-8')
                with open(os.path.join(output_path, short_name + '.tex'),'w')\
                        as fname:
                    fname.write(decoded_str)
            except UnicodeDecodeError:
                # file_gz is not zipped so try to untar the file instead
                print('attempting to write %s to %s'%(short_name,output_path))
                with tarfile.open(self.tar_path) as ff:
                    tar2 = ff.extractfile(filename)
                    with tarfile.open(fileobj=tar2) as tars:
                        tars.extractall(path=output_path)
            # the file 1804/1804.01815.gz in 1804_005 has something different

        return True

    def extract_str(self, filename):
        '''
        given filename, extract a file with that name from
        self.tar
        '''
        with tarfile.open(self.tar_path) as ff:
            file_gz = ff.extractfile(filename)
            with gzip.open(file_gz,'rb') as fb:
                file_str = fb.read()
        return file_str.decode('utf-8')


if __name__ == '__main__':
    x = Xtraction(sys.argv[1])
    f_lst = x.filter_MSC('math.DG')
    for f in f_lst:
        print('writing file %s'%f)
        x.extract_any(f, sys.argv[2])
        #with open(os.path.join(sys.argv[2],tar2api(f) + '.tex'), 'w') as fname:
        #    fname.write(x.extract_dir(f))
    
