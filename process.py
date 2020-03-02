import sys
sys.path.insert(0,'arxiv.py/')
import arxiv
import tarfile
import gzip
import re
import os.path
import chardet
import sqlalchemy as sa
import magic
import databases.create_db_define_models as cre

commentary_filename = 'latexml_commentary.txt'

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
    >>> api2tar('http://arxiv.org/abs/1804.00018v1',1)
    '1804/1804.00018'
    >>> api2tar('http://arxiv.org/abs/1804.0018v1',1)
    '1804/1804.0018'
    >>> api2tar('http://arxiv.org/abs/math/0303001v7',2)
    '0303/math0303001'
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
    >>> detect_format('1701/1701.00253.gz')
    (1, '[0-9]{4}/([0-9]{4})\\\\.([0-9]{4,5})')
    >>> detect_format('0701/astro-ph0701321.gz')
    (2, '[0-9]{4}/([a-z\\\\-]+)([0-9]{7}).+')
    '''
    regex1 = r'[0-9]{4}/([0-9]{4})\.([0-9]{4,5})'
    regex2 = r'[0-9]{4}/([a-z\-]+)([0-9]{7}).+'
    if re.match(regex1, id_str):
        return 1, regex1
    elif re.match(regex2, id_str):
        return 2, regex2
    else:
        return None


def tar2api(id_str, **kwargs):
    '''
    The name of the files when untared has the format:
    '1804/1804.00020.gz', '1804/1804.00239.gz', '1804/1804.00127.gz', '1804/1804.00197.gz'
    in order to get a searcheable id in the API take the leading number and the file type
    >>> tar2api('1804/1804.00239.gz')
    '1804.00239'
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
    >>> tar2api2('0703/astro-ph0703001.gz')
    'astro-ph/0703001'
    '''
    name = re.match(r'[0-9]{4}/([a-z\-]+)([0-9]{7})', id_str)
    if name:
        return name.group(1) + sep + name.group(2)
    else:
        raise Exception('No results found in string: %s'%id_str)

def Tar2api(id_str, sep='/'):
    '''
    This function should match both tar2api and tar2api2
    >>> Tar2api('1804/1804.00239.gz', sep='.')
    '1804.00239'

    >>> Tar2api('0703/astro-ph/0703001.gz')
    'astro-ph/0703001'

    >>> Tar2api('0703/astro-ph/0703001.gz', sep='.')
    'astro-ph.0703001'
    '''
    #Trying to match either 
    regex1 = r'.+/([0-9]{4})\.([0-9]{4,5})'
    # want to catch http://arxiv.org/abs/math/9212204v1
    regex2 = r'.+/([a-z\-]+)/([0-9]{7}).+'
    if re.match(regex1, id_str):
        #format_detected  '1804/1804.00239.gz'
        name = re.match(r'.*([0-9]{4})\.([0-9]{4,5})',id_str)
        try:
            return_str = name.group(1) + '.' + name.group(2)
        except AttributeError:
            raise ValueError('Error: tar2api cannot find results in string: %s'%id_str)
        return return_str
    elif re.match(regex2, id_str):
        # format detected http://arxiv.org/abs/math/9212204v1
        name = re.match(r'.+/([a-z\-]+)/([0-9]{7})', id_str)
        if name:
            return name.group(1) + sep + name.group(2)
        else:
            raise ValueError('No results found in string: %s'%id_str)
    else:
        raise ValueError('The format of the id_str: %s was not matched by any known format'%id_str)

def tar_id(id_str):
    '''
    Given a path like: /mnt/arXiv_src/src/arXiv_src_1804_001.tar
    return the unique identifier 1804_001
    >>> tar_id('/mnt/arXiv_src/src/arXiv_src_1804_001.tar')
    '1804_001'
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

def sliced_article_query(article_lst, slice_length=300):
    '''
    query the arxiv metadata API using the arxiv.py library
    '''
    query_results = []
    i = 0
    sl = slice_length # slice length
    while i*sl <= len(article_lst):
        second_length = min((i + 1)*sl, len(article_lst))
        query_results +=\
                arxiv.query(id_list=article_lst[i*sl:second_length],
                max_results=(sl + 1))
        print('querying from %02d to %02d successful       \r'\
                %(i*sl,second_length),end='\r')
        i += 1
    return query_results


class Xtraction(object):
    def __init__(self, tar_path, *argv, **kwarg):
        #get a list of all objects in the file at tar_path
        #these have name format: '1804/1804.00020.gz'
        # if db option is given, then use database and don't query the arXiv
        if os.path.isfile(tar_path):
            self.tar_path = tar_path
        else:
            raise(ValueError('There is no file at %s'%tar_path))
        print('untaring file: %s    \r'%tar_path,end='\r')
        with tarfile.open(tar_path) as fa:
            self.art_lst = [k.get_info()['name'] for k in fa.getmembers()][1:]

        self.format_found, self.format_regex = detect_format(self.art_lst[1])
        if self.format_found == 1:
            query_id_list = list(map(tar2api, self.art_lst))
            self.tar2api = tar2api
            self.api2tar = lambda name : api2tar(name,1)
        elif self.format_found == 2:
            query_id_list = list(map(tar2api2, self.art_lst))
            self.tar2api = tar2api2
            self.api2tar = lambda name : api2tar(name,2)
        else:
            raise Exception('Could not determine format of %s'%self.art_lst[:5])

        # check if a database was given
        self.db_path = kwarg.get('db', None)
        if self.db_path:
            basename = self.tar_path.split('/')[-1]
            right_name = 'src/' + basename
            engine = sa.create_engine(self.db_path, echo=False)
            engine.connect()
            SMaker = sa.orm.sessionmaker(bind=engine)
            session = SMaker()
            q = session.query(cre.ManifestTarFile)
            resu = q.filter_by(filename = right_name)
            foreign_key_id = resu.first().id

            Q_lst = session.query(cre.Article).filter(cre.Article.tarfile_id == foreign_key_id).all()
            self.query_results = []
            for q in Q_lst:
                q_dict = {}
                q_dict['id'] = q.id
                q_dict['tags'] = eval(q.tags)
                q_dict['arxiv_primary_category'] = q_dict['tags'][0]
                self.query_results.append(q_dict)

        else:
            print("\033[K",end='')
            print('querying the arxiv \r',end='\r')
            #  query with the arxiv API and arxiv package
            self.query_results = sliced_article_query(query_id_list)


        self.encoding_dict = {
         'utf-8':  ['utf-8',],
         'ascii':  ['utf-8',],
        'ISO-8859-1': ['ISO-8859-1',],
        'Windows-1252': ['latin1',],
         'SHIFT_JIS': ['shift_jis',],
         'GB2312': ['gbk', 'gb18030-2000'],
         'Big5':  ['big5', 'gbk'],
        'windows-1251': ['windows-1251',],
        'IBM866': ['ibm866', ],
        'IBM855': ['ibm855', ],
        'Windows-1254': ['cp932', 'latin1'],
        'TIS-620': ['cp874', ],
        'HZ-GB-2312': ['gbk', ],
        'KOI8-R': ['koi8-r', ],
        'MacCyrillic': ['koi8-r', ],
        'EUC-JP': ['euc_jp', ],
        'EUC-KR': ['euc_kr', ],
        'ISO-2022-JP': ['iso2022_jp', 'cp932', ],
        'windows-1255': ['cp1255', ],
        'CP949': ['cp949', ],
        None: None,
        }

    def filter_MSC(self, MSC, run_api2tar=True):
        if run_api2tar:
            return [api2tar(d.id, self.format_found) 
                    for d in self.query_results if d['tags'][0]['term']==MSC]
        else:
            return [d for d in self.query_results if d['tags'][0]['term']==MSC]

    def filter_arxiv_meta(self, *args):
        '''
        return a list of the name of all the tar file member names ex. 0303/math0303004.gz
        whose tag value contains the string name
        ** This function assumes that all the query results from the API
        have an arxiv_primary_category entry will fail otherwise
        '''
        ind_lst = []
        for ind, q in enumerate(self.query_results):
            tag_value = q['arxiv_primary_category']['term']
            if any([term in tag_value for term in args]):
                ind_lst.append(ind)

        return [self.art_lst[index] for index in ind_lst]

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
        use encoding_dict to and chardet to try decode it and 
        annotate the encoding used
        '''
        encoding_detected = chardet.detect(file_str)['encoding']
        encoding_lst = self.encoding_dict.get(encoding_detected, 'Unk')
        commentary_dict = {}
        commentary_dict['encoding detected'] = encoding_detected
        if encoding_lst == 'Unk':
            raise ValueError('Unknown encoding %s found in file %s\
                    in tarfile %s'%(encoding_detected, filename,
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
                decoded_str = file_str.decode(errors='ignore')
                commentary_dict['decode_error'] = 'tried %s on file %s\
                        but all failed'%(str(encoding_lst), filename)
        else:
            # If no codec was detected just 
            # ignore the problem :( and use the default (utf-8)
            comm_mess = 'Unknown encoding: %s\
                    in file: %s decoding with utf8'%(encoding_detected,
                            filename)
            commentary_dict['decode_message'] = comm_mess
            decoded_str = file_str.decode()
        return decoded_str, commentary_dict


    def extract_tar(self, output_dir, *args, **kwargs):
        '''
        Extract the file in self.tarfile to output_dir
        Optional argument 'term': string with term in the arxiv_primary_category to extract
        Examples for term: `math.AG`, `math` , `physics` etc

        keyword article_name = r'string' 
        where string is a regular expresion contained in the self.art_lst
        Ex. article_name=r'^1703\.0137.*' (DON'T FORGET THE R AT THE START)
        This option overrides all others

        '''
        if args == () or 'all' == args[0]:
            loop_filenames = self.art_lst
        else:
            loop_filenames = self.filter_arxiv_meta(*args)

        art_name = kwargs.get('article_name', None)
        if art_name:
            # Select the name of articles in the tar file that contain a art_name as a subtring
            loop_filenames = [name for name in self.art_lst if re.match(art_name, name)]

        ff = tarfile.open(self.tar_path) #open the .tar file once
        for filename in loop_filenames:
            print("\033[K",end='') 
            print('writing file %s               \r'%filename, end='\r')

            short_name = self.tar2api(filename, sep='.')

            commentary_dict = { 'tar_file': os.path.basename(self.tar_path) }
            output_path = os.path.join(self.path_dir(output_dir), short_name)
            os.mkdir(output_path)

            # If file is pdf we don't care about it
            if '.pdf' in filename:
                out_mess = 'pdf file, omitting %s'%filename
                commentary_dict['file_error'] = out_mess
            else:
                file_gz = ff.extractfile(filename)
                gz_magic = magic.detect_from_content(file_gz.read(2048))
                file_gz.seek(0)

                # With the magic info of the file we can tell if it is pdf only or .cry encrypted
                # TODO improve this with a regex
                if '.tex.cry"' in gz_magic.name:
                    out_mess = '.tex.cry file found %s'%filename 
                    commentary_dict['file_error'] = out_mess

                else:
                    with gzip.open(file_gz,'rb') as fgz:
                        snd_magic = magic.detect_from_content(fgz.read(2048))
                        fgz.seek(0)
                        # if the filename stands for a directory
                        if snd_magic.mime_type == 'application/x-tar':
                            with tarfile.open(fileobj=fgz) as fb:
                                fb.extractall(path=output_path)
                                commentary_dict['extraction_tool'] = 'tarfile'
                        else:
                        # the file is not a tar so try to decode it
                            try:
                                file_str = fgz.read()
                                decoded_str, comm_dict = self.decoder(file_str,
                                        filename)
                                commentary_dict = {**commentary_dict,**comm_dict}
                            except UnicodeDecodeError as ee:
                                commentary_dict['decode_error'] = str(ee)
                                decoded_str = 'Empty file goes here'

                            with open(os.path.join(output_path,
                                short_name + '.tex'),'w') as fname:
                                fname.write(decoded_str)
            write_dict(commentary_dict, os.path.join(output_path, commentary_filename))
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
            # format 0703/math0703071.gz
            short_name = tar2api2(filename, sep='.')
        else:
            raise Exception('short_name will not be defined because no format was found')
        commentary_dict = { 'tar_file': os.path.basename(self.tar_path) }
        output_path = os.path.join(self.path_dir(output_dir), short_name)
        os.mkdir(output_path)
        ff = tarfile.open(self.tar_path)
        try:
            file_gz = ff.extractfile(filename+'.gz')
        except KeyError:
            commentary_dict['KeyError'] = 'Check if file %s is pdf only'%filename
            write_dict(commentary_dict, os.path.join(output_path, commentary_filename))
            return True

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
        # Gunzip file
            with gzip.open(file_gz,'rb') as fgz:
                file_str = fgz.read()
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
                try:
                    decoded_str = file_str.decode()
                except UnicodeDecodeError:
                    commentary_dict['decode_failed'] = "Possible encrypted file (.cry) found."
                    decoded_str = 'Empty file goes here'

        #raise ValueError('The file: %s has an unknown encoding: %s. fix it!'%(filename, encoding_detected))
            with open(os.path.join(output_path, short_name + '.tex'),'w')\
                    as fname:
                fname.write(decoded_str)
        ff.close()

        write_dict(commentary_dict, os.path.join(output_path, commentary_filename))
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

    def save_articles_to_db(self, database):
        '''
        Save all the articles from query_results to 
        the database
        Example: x.save_articles_to_db('sqlite:///arxiv1.db')
        where x is an Xtraction instance
        '''
        #Get the basename
        basename = self.tar_path.split('/')[-1]
        right_name = 'src/' + basename
        engine = sa.create_engine(database, echo=False)
        engine.connect()
        SMaker = sa.orm.sessionmaker(bind=engine)
        session = SMaker()
        q = session.query(cre.ManifestTarFile)
        resu = q.filter_by(filename = right_name)
        foreign_key_id = resu.first().id
        #assert(resu.count() == 1, 'Filename %s is not unique'%right_name)
        #return resu.first().id
        session.add_all([cre.new_article_register(D, foreign_key_id)\
                for D in self.query_results])
        session.commit()


if __name__ == '__main__':
    '''
    usage python[3] process.py tar_file_path... outdir [--term math.AG...] [--db database]

    The optional database to use instead of downloading with the API
    '''
    import argparse
    parser = argparse.ArgumentParser(description='parsing xml commandline script')
    parser.add_argument('tarpath', type=str, nargs='+',
            help='Path to the arXiv source tar file ex. arXiv_src_1804_004.tar')
    parser.add_argument('outdir', type=str,
            help='Path to the arXiv source tar file ex. arXiv_src_1804_004.tar')
    parser.add_argument('--term', type=str, nargs='+', default=['all'],
            help='terms to filter out ex. math.AG, math.AP')
    parser.add_argument('--db', type=str, default=None,
            help='path of the database to get the metadata do not include the sqlite:/// part')
    args = parser.parse_args(sys.argv[1:])

    print('Tar paths are:', args.tarpath)
    print('Outdir is:    ', args.outdir)
    print('Terms are:    ', args.term)
    print('Database path is:', args.db)

    for T in args.tarpath:
        if args.db:
            X = Xtraction(T, db='sqlite:///' + args.db)
        else:
            X = Xtraction(T)
        print("\033[K",end='')
        print('writing file %s               \r'%T,end='\r')
        X.extract_tar(args.outdir, *(args.term))
