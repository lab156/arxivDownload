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
    with open(filename, 'a') as fw:
        fw.write(''.join([str(d) + ': ' + str(dic[d]) +'\n' for d in dic]))

def api2tar(id_str,format_int):
    '''
    The arxiv API produces ids in the format:
     format1:   http://arxiv.org/abs/1804.00018v1',
     format2:   http://arxiv.org/abs/math/0703021v1
    The tar files have name format:
   format1: <TarInfo '1804/1804.00020.gz' at 0x7f0ba68c8cc8>
   format2: 0701/cond-mat0701697.pdf',
    '0701/cond-mat0701698.gz',
     '0701/cond-mat0701699.gz',
      '0701/cond-mat0701700.gz',
       '0701/cond-mat0701701.gz',
    '''
    try:
        if format_int == 1:
            name = re.match(r'.*([0-9]{4})\.([0-9]{4,5})',id_str)
            return_str = name.group(1) + '/' + name.group(1) + '.' + name.group(2)
        elif format_int == 2:
            name = re.match(r'http://arxiv.org/abs/([a-z\-]+)/([0-9]{4})([0-9]{3})',id_str)
            return_str = name.group(2) + '/' + name.group(1) + name.group(2) + name.group(3)
    except AttributeError:
        raise Exception('no results for id_str: %s '%id_str)

    return return_str

def detect_format(id_str):
    '''
    The tar files of the bulk download come in different formats
    format 1:      1701/1701.00253.gz 
    format 2:     0701/astro-ph0701321.gz
    When id_str is one of these formats return the format number
    '''
    regex1 = r'[0-9]{4}/([0-9]{4})\.([0-9]{4,5})'
    regex2 = r'[0-9]{4}/([a-z\-]+)([0-9]{7}).+'
    if re.match(regex1, id_str):
        return 1, regex1
    elif re.match(regex2, id_str):
        return 2, regex2
    else:
        return None


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

def tar2api2(id_str, sep='/'):
    '''
    Old tar files have the format:
    0703/astro-ph0703001.gz
    it was used until about march 03, 2007
    to search arxiv.org it has to be put in the format
    astro-ph/0703001
    '''
    name = re.match(r'[0-9]{4}/([a-z\-]+)([0-9]{7})', id_str)
    if name:
        return name.group(1) + sep + name.group(2)
    else:
        raise Exception('No results found in string: %s'%id_str)
    

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
        with tarfile.open(tar_path) as fa:
            self.art_lst = [k.get_info()['name'] for k in fa.getmembers()]

        self.format_found, self.format_regex = detect_format(self.art_lst[1])
        if self.format_found == 1:
            query_id_list = list(map(tar2api, self.art_lst[1:]))
        elif self.format_found == 2:
            query_id_list = list(map(tar2api2, self.art_lst[1:]))
        else:
            raise Exception('Could not determine format of %s'%self.art_lst[:5])

        print("\033[K",end='') 
        print('querying the arxiv \r',end='\r')
        self.query_results = []
        i = 0
        sl = 300 # slice length 
        while i*sl < len(self.art_lst):
            second_length = min((i + 1)*sl, len(self.art_lst))
            try: 
                self.query_results += arxiv.query(id_list=query_id_list[i*sl:second_length],
                        max_results=(sl + 1))
                print('querying from %02d to %02d successful       \r'%(i*sl,second_length),end='\r')
            except Exception:
                print('query of %s unsuccessful       \r'%os.path.basename(self.tar_path),end='\r')
                break
            i += 1

        self.encoding_dict = {
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

    def filter_MSC(self, MSC , run_api2tar=True):
        if run_api2tar:
            return [api2tar(d.id, self.format_found) for d in self.query_results \
                    if d['tags'][0]['term']==MSC]
        else:
            return [d for d in self.query_results if d['tags'][0]['term']==MSC]

    def path_dir(self, output_dir):
        '''
        Create if not exists dir with path:
        output_dir
          |_ 1804.0001   (tar_id name)
        '''
        #the_path = os.path.join(output_dir, tar_id(self.tar_path))
        the_path = output_dir
        if not os.path.exists(the_path):
            os.makedirs(the_path)
        return the_path

    def decoder(self, file_str, filename=''):
        '''
        Given file_str a raw binary string with unknown encoding,
        use encoding_dict to and chardet to try decode it and annotate the encoding used
        '''
        encoding_detected = chardet.detect(file_str)['encoding']
        encoding_lst = self.encoding_dict.get(encoding_detected, 'Unk')
        commentary_dict = {}
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
                commentary_dict['decode_message'] = 'tried %s on file %s but all failed'%(str(encoding_lst),
                        filename)

        else:
            # If no codec was detected just ignore the problem :( and use the default (utf-8)
            comm_mess = 'Unknown encoding: %s in file: %s decoding with utf8'%(encoding_detected,
                                                                               filename)
            commentary_dict['decode_message'] = comm_mess
            decoded_str = file_str.decode()
        return decoded_str, commentary_dict

    def extract_tar(self, output_dir, term):
        '''
        Extract the file in self.tarfile to output_dir
        '''
        f_lst = self.filter_MSC(term) 
        ff = tarfile.open(self.tar_path) #open the .tar file once
        for filename in f_lst:
            print("\033[K",end='') 
            print('writing file %s               \r'%filename, end='\r')
            if self.format_found == 1:
                short_name = tar2api(filename) # format 1804.00000
            elif self.format_found == 2:
                short_name = tar2api2(filename, sep='.') # format 0703/math0703071.gz turn into math.0703071
            else:
                raise Exception('short_name will not be defined because no format was found')
            commentary_dict = { 'tar_file': os.path.basename(self.tar_path) }
            output_path = os.path.join(self.path_dir(output_dir), short_name)
            os.mkdir(output_path)
            try:
                file_gz = ff.extractfile(filename + '.gz')
                with gzip.open(file_gz,'rb') as fgz:
                    file_str = fgz.read()
                try:
                    # if the subtar is another tar then extractfile will be successfull
                    with tarfile.open(self.tar_path) as fb:
                        tar2 = fb.extractfile(filename + '.gz')
                        with tarfile.open(fileobj=tar2) as tars:
                            tars.extractall(path=output_path)
                            commentary_dict['extraction_tool'] = 'tarfile'
                except tarfile.ReadError:   
                    # this means tarfile is not a tar so we try to decode it
                    decoded_str, comm_dict = self.decoder(file_str, filename)
                    write_dict(comm_dict, os.path.join(output_path, 'commentary.txt'))
                    with open(os.path.join(output_path, short_name + '.tex'),'w')\
                            as fname:
                        fname.write(decoded_str)
            except KeyError:
                #if the file is .pdf there is no tex and we don't care about it
                matching_filenames = [n for n in self.art_lst if filename in n]
                out_mess = 'Check if file %s is pdf only'%matching_filenames 
                commentary_dict['KeyError'] = out_mess
                #return True
            write_dict(commentary_dict, os.path.join(output_path, 'commentary.txt'))
        ff.close()
        print('successful extraction of  %s      '%os.path.basename(self.tar_path))
        return True




    def extract_any(self, filename, output_dir):
        '''
        given filename, decide it is a tape archive (tar)
        or gzipped compressed file and extract it to the directory output_dir
        self.tar
        Example
        filename: /mnt/arXiv_src/src/arXiv_src_1804_001.tar
        output_dir: math.DG
        '''
        if self.format_found == 1:
            short_name = tar2api(filename) # format 1804.00000
        elif self.format_found == 2:
            short_name = tar2api2(filename, sep='.') # format 0703/math0703071.gz
        else:
            raise Exception('short_name will not be defined becuase no format was found')
        commentary_dict = { 'tar_file': os.path.basename(self.tar_path) }
        output_path = os.path.join(self.path_dir(output_dir), short_name)
        os.mkdir(output_path)
        ff = tarfile.open(self.tar_path) 
        try:
            file_gz = ff.extractfile(filename+'.gz')
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
            with tarfile.open(self.tar_path) as fb:
                tar2 = fb.extractfile(filename + '.gz')
                with tarfile.open(fileobj=tar2) as tars:
                    #print('extracting tar file %s to %s'\
                    #        %(short_name,output_path))
                    tars.extractall(path=output_path)
                    commentary_dict['extraction_tool'] = 'tarfile'
        except tarfile.ReadError:
        # if reading the tarfile fails then it must be compressed file
        # detecting the encoding first is very slow
            encoding_detected = chardet.detect(file_str)['encoding']
            encoding_lst = self.encoding_dict.get(encoding_detected, 'Unk')
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
        with tarfile.open(self.tar_path) as fc:
            file_gz = fc.extractfile(filename)
            with gzip.open(file_gz,'rb') as fb:
                file_str = fb.read()
        return file_str.decode('utf-8')

if __name__ == '__main__':
    file_lst = sys.argv[1:-1]
#    x = Xtraction(sys.argv[1])
#    x.extract_tar(sys.argv[-1], 'math.AG')

    for f_path in file_lst:
        print('starting extraction of  %s         \r'%os.path.basename(f_path),end='\r')
        x = Xtraction(f_path)
        f_lst = x.filter_MSC('math.AG')
        for f in f_lst:
            print("\033[K",end='') 
            print('writing file %s               \r'%f,end='\r')
            x.extract_any(f, sys.argv[-1])
        print('successful extraction of  %s      '%os.path.basename(f_path))
