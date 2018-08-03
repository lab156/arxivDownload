import sys
sys.path.insert(0,'arxiv.py/')
import arxiv
import tarfile
import gzip
import re
import os.path
import chardet

def write_dict(dic, filename):
    '''
    pretty print the commentary dictionary
    '''
    with open(filename, 'a') as ff:
        ff.write(''.join([str(d) + ': ' + str(dic[d]) +'\n' for d in dic]))

def api2tar(id_str):
    '''
    The arxiv API produces ids in the format:
    http://arxiv.org/abs/1804.00018v1',
 http://arxiv.org/abs/1804.00193v1',
 http://arxiv.org/abs/1804.00259v2
    The tar files have name format:
    <TarInfo '1804/1804.00020.gz' at 0x7f0ba68c8cc8>
    '''
    name = re.match(r'.*([0-9]{4})\.([0-9]{4,5})',id_str)
    return name.group(1) + '/' + name.group(1) + '.' + name.group(2) + '.gz'

def tar2api(id_str):
    '''
    The name of the files when untared has the format:
    '1804/1804.00020.gz', '1804/1804.00239.gz', '1804/1804.00127.gz', '1804/1804.00197.gz'
    in order to get a searcheable id in the API take the leading number and the file type
    '''
    name = re.match(r'.*([0-9]{4})\.([0-9]{4,5})',id_str)
    try:
        return_str = name.group(1) + '.' + name.group(2) 
    except AttributeError:
        print('Error: tar2api cannot find results in string: %s'%id_str)
        raise ValueError
        
    return return_str

def tar_id(id_str):
    '''
    Given a path like: /mnt/arXiv_src/src/arXiv_src_1804_001.tar
    return the unique identifier 1804_001
    '''
    name = re.match(r'.*arXiv_src_([0-9]{4})_([0-9]{3})\.tar', id_str)
    return name.group(1) + '_' + name.group(2)

def year(id_str):
    '''
    attempts to guess the year name from the common strings.
    searches in id_str for 4 consecutive digits and returns the first 2
    '''
    name = re.match(r'.*([0-9]{4}).*', id_str)
    return name.group(1)[:2]
    

    
class Xtraction(object):
    def __init__(self, tar_path, *argv, **kwarg):
        #get a list of all objects in the file at tar_path
        #these have name format: '1804/1804.00020.gz'
        if os.path.isfile(tar_path):
            self.tar_path = tar_path
        else:
            raise(ValueError('There is no file at %s'%tar_path))
        print('untaring file: %s    \r'%tar_path,end='\r')
        with tarfile.open(tar_path) as ff:
            self.art_lst = [k.get_info()['name'] for k in ff.getmembers()]

        #To query an id, from  '1804/1804.00020.gz' we just get 1804.00020
        query_id_list = [tar2api(s) for s in self.art_lst[1:]]

        print("\033[K",end='') 
        print('querying the arxiv \r',end='\r')
        self.query_results = []
        i = 0
        sl = 500 # slice length 
        while i*sl < len(self.art_lst):
            second_length = min((i + 1)*sl, len(self.art_lst))
            try: 
                self.query_results += arxiv.query(id_list=query_id_list[i*sl:second_length],
                        max_results=(sl + 1))
                print('query of %s successful       \r'%os.path.basename(self.tar_path),end='\r')
            except Exception:
                print('query of %s unsuccessful       \r'%os.path.basename(self.tar_path),end='\r')
                break
            i += 1

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
        #the_path = os.path.join(output_dir, tar_id(self.tar_path))
        the_path = output_dir
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
        short_name = tar2api(filename) # format 1804_00000
        commentary_dict = { 'tar_file': os.path.basename(self.tar_path) }
        output_path = os.path.join(self.path_dir(output_dir), short_name)
        os.mkdir(output_path)
        ff = tarfile.open(self.tar_path) 
        try:
            file_gz = ff.extractfile(filename)
        except KeyError:
            out_mess = 'Check if file %s is pdf only'%filename 
            #print(out_mess)
            commentary_dict['KeyError'] = out_mess
            write_dict(commentary_dict, os.path.join(output_path, 'commentary.txt'))
            return True
        #if tarfile.is_tarfile(file_gz):
#                with tarfile.open(file_gz) as ftar: 
#                    ftar.extractall(output_path)
#            else:
        with gzip.open(file_gz,'rb') as fgz:
            file_str = fgz.read()
        try:
            with tarfile.open(self.tar_path) as ff:
                tar2 = ff.extractfile(filename)
                with tarfile.open(fileobj=tar2) as tars:
                    #print('extracting tar file %s to %s'\
                    #        %(short_name,output_path))
                    tars.extractall(path=output_path)
                    commentary_dict['extraction_tool'] = 'tarfile'
        except tarfile.ReadError:
        # if reading the tarfile fails then it must be compressed file
        # detecting the encoding first is very slow
            encoding_detected = chardet.detect(file_str)['encoding']
            encoding_dict = {
             'utf-8':  ['utf-8',],
             'ascii':  ['utf-8',],
            'ISO-8859-1': ['ISO-8859-1',],
            'Windows-1252': ['latin1',],
             'SHIFT_JIS': ['shift_jis',],
             'GB2312': ['gbk', 'gb18030-2000'],
             'Big5':  ['big5',],
            'windows-1251': ['windows-1251',],
            'IBM866': ['ibm866', ],
            'Windows-1254': ['cp932', 'latin1'],
            'TIS-620': ['cp874', ],
            'HZ-GB-2312': ['gbk', ],
            'KOI8-R': ['koi8-r', ],
            'EUC-JP': ['euc_jp', ],
            'ISO-2022-JP': ['iso2022_jp',],
            None: None,
            }
            encoding_lst = encoding_dict.get(encoding_detected, 'Unk')
            commentary_dict['encoding detected'] = encoding_detected
            if encoding_lst == 'Unk':
                raise ValueError('Unknown encoding %s found in file %s in tarfile %s'%(encoding_detected,
                                                                                      filename,
                                                                            os.path.basename(self.tar_path)))
            elif encoding_lst:
                i = 0
                while i < len(encoding_lst):
                    try:
                        decoded_str = file_str.decode(encoding_lst[i])
                        break
                    except UnicodeDecodeError:  
                        i += 1
                else:
                    print('tried %s on file %s but all failed'%(str(encoding_lst), filename))
            else:
                # If no codec was detected just ignore the problem :(
                comm_mess = 'Unknown encoding: %s in file: %s decoding with utf8'%(encoding_detected, filename)
                #print(comm_mess)
                commentary_dict['decode_message'] = comm_mess
                decoded_str = file_str.decode()
                #raise ValueError('The file: %s has an unknown encoding: %s. fix it!'%(filename, encoding_detected))
                
            with open(os.path.join(output_path, short_name + '.tex'),'w')\
                    as fname:
                fname.write(decoded_str)
        ff.close()

        write_dict(commentary_dict, os.path.join(output_path, 'commentary.txt'))
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
    file_lst = sys.argv[1:-1]
    for f_path in file_lst:
        print('starting extraction of  %s         \r'%os.path.basename(f_path),end='\r')
        x = Xtraction(f_path)
        f_lst = x.filter_MSC('math.AG')
        for f in f_lst:
            print("\033[K",end='') 
            print('writing file %s               \r'%f,end='\r')
            x.extract_any(f, sys.argv[-1])
        print('successful extraction of  %s      '%os.path.basename(f_path))
   
