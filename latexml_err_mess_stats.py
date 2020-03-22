import datetime as dt
import re
import dateutil.parser as duparser
import os
import enum
import numpy as np
import pandas as pd
import collections as coll
import magic
import tarfile
import logging

commentary_filename = 'latexml_commentary.txt'

def parse_conversion(f_str):
    """
    parses the conversion line in the latexml_errors_mess.txt file
    """
    return re.search('\\nConversion complete: (No obvious problems\.)?'
    '(\d+ warnings?[;\.] ?)?'
    '(\d+ errors?[;\.] ?)?'
    '(\d+ fatal errors?[;\.] ?)?'
    '(\d+ undefined macros?\[[\*\@\{\}\\\\,\w\. ]+\][;\.] ?)?'
    '(\d+ missing files?\[[,\w\. ]+\])?.*\n', f_str).groups()

def entry_handler(str_or_none):
    '''
    deals with entries of parse_conversion

    >>> entry_handler('123 warnings')
    123
    >>> entry_handler(None)
    0
    '''
    if str_or_none:
        return int(str_or_none.split()[0])
    else:
        return 0

class Result(enum.Flag):
    SUCC = enum.auto()  # everything went fine as far as I can tell
    TIMED = enum.auto() # timed out more than the allowed time in configuration
    FATAL = enum.auto() # fatal error found
    MAXED = enum.auto() # maxed out the allowed number of errors
    DIED = enum.auto() # found dead: ex. no finished processing timestamp
    NOTEX = enum.auto() # No TeX file was found it might be pdf only or a weird case like 1806.03429
    #NOLOG = enum.auto() # No log file, different from DIED and NOTEX
    FAIL = FATAL | TIMED | MAXED | DIED

fun_dict = { 'success' : lambda p: Result.SUCC in p.result,
        'fail' : lambda p : p.result in Result.FAIL,
        'fatal' : lambda p : Result.FATAL in p.result,
        'maxed' : lambda p :  Result.MAXED in p.result,
        'timed' : lambda p : Result.TIMED in p.result,
        'died' : lambda p : Result.DIED in p.result,
        'notex' : lambda p : Result.NOTEX == p.result,}

class ParseLaTeXMLLog():
    def __init__(self, error_log, commentary, article_name, max_errors=10000):
        '''
        Common actions to do on a latexml errors messages log
        log_path is the path to a latexml_err_mess_ log file
        can be a .tar file also 
        files need to be bufferedIO objects so open them up with the rb (b stands for binary) option
        '''
        ## Error_log can be none but the commentary file is necessary
        self.commentary = list(map(lambda x: x.decode(), commentary.readlines()))
        self.filename = article_name
        if error_log:
            self.log = error_log.read().decode()


        #if os.path.isfile(log_path):
        #    self.filename = log_path
        #    self.dir_name = os.path.split(self.filename)[0]
        #else: # log_path is a directory
        #    self.filename = os.path.join(log_path, 'latexml_errors_mess.txt')
        #    self.dir_name = log_path

#        if os.path.isfile(self.filename):
#            with open(self.filename, 'r') as log_fobj:
#                self.log = log_fobj.read()

            # Get the time span
            try:
                self.start = re.search('\\nprocessing started (.*)\\n',
                        self.log).group(1)
            except AttributeError:
                import pdb; pdb.set_trace()
            try:
                self.finish = re.search('\\nprocessing finished (.*)\\n',
                        self.log).group(1)
            except AttributeError:
                self.result = Result.DIED
                self.warnings = self.errors =\
                        self.fatal_errors = self.undefined_macros =\
                        self.missing_files = self.no_prob = np.NAN
                self.time_secs = np.NAN
            else:
                d1 = duparser.parse(self.start)
                d2 = duparser.parse(self.finish)
                self.time_secs = (d2-d1).seconds

                # Get the conversion stats
                conversion_tuple = parse_conversion(self.log)
                self.warnings = entry_handler(conversion_tuple[1])
                self.errors = entry_handler(conversion_tuple[2])
                self.fatal_errors = entry_handler(conversion_tuple[3])
                self.undefined_macros = entry_handler(conversion_tuple[4])
                self.missing_files = entry_handler(conversion_tuple[5])
                self.no_prob = conversion_tuple[0]

            if self.fatal_errors == 0:
                self.result = Result.SUCC
            else:
                if self.errors > max_errors:
                    self.result = Result.MAXED
                else:
                    self.result = Result.FATAL
                if self.timedout():
                    self.result |= Result.TIMED

        else:
            # TODO: need to break down why there is no error_log
            self.result = Result.NOTEX
            #self.result += Result.NOLOG
            self.time_secs = np.NAN


    def commentary(self):
        '''
        attempts to read the latexml_commentary.txt file in a latexml processed directory
        '''
        with open(os.path.join(self.dir_name, commentary_filename), 'r') as commentary_fobj:
            comm_str = commentary_fobj.readlines()
        return comm_str

    def get_encoding(self):
        '''
        Tries to get the encoding from the latexml_commentary files
        '''
        find_encod = lambda s: re.search(r'encoding detected: (.*)$', s)
        encod_map = map(find_encod, self.commentary)
        encod_lst = [e.group(1) for e in encod_map if e is not None]

        if  any(encod_lst):
            encod = encod_lst[0]
        else:
            encod = None
        return encod




    def finished(self):
        '''
        return time if process timed out
        return None if process finished on time 
        '''
        result = re.search('Finished in less than (\d+) seconds', self.commentary[-1])

        if result:
            return int(result.group(1))
        else:
            return None

    def timedout(self):
        '''
        return time if the LAST LINE of the commentary file says it timed out
        return None if process finished on time
        '''
        result = re.search('Timeout of (\d+) seconds occured', self.commentary[-1])

        if result:
            return int(result.group(1))
        else:
            return None

    def flag(self, max_errors = 100):
        '''
        true if exceeded the number of allowed errors
        '''
        temp_result = None

commentary_pred = lambda x: 'latexml_commentary' in x
error_log_pred = lambda x: 'latexml_errors' in x

def open_tar(tarpath, **kwargs):
    '''
     `tarpath` is the path of tar file in the specific format
     '''
    print_opt = kwargs.get('print', None)
    pvec = np.zeros(7)
    encoding_lst = []
    times_lst = []
    article_dict = coll.defaultdict(list)
    with tarfile.open(tarpath) as tar_file:
        for pathname in tar_file.getnames():
            dirname = pathname.split('/')[1]
            article_dict[dirname].append(pathname)
        for name,val in article_dict.items():
            comm = tar_file.extractfile(next(filter(commentary_pred, val)))
            log_name = next(filter(error_log_pred, val), None)
            if log_name:
                log = tar_file.extractfile(log_name)
            else:
                log = None
            #print(log, ' ', comm, ' ')
            p = ParseLaTeXMLLog(log, comm, name)
            assert hasattr(p, 'time_secs'), " Error, %s has no attribute time_secs"%p.filename
            pvec += (fun_dict['success'](p),
                fun_dict['fail'](p),
                fun_dict['fatal'](p),
                fun_dict['maxed'](p),
                fun_dict['timed'](p),
                fun_dict['died'](p),
                fun_dict['notex'](p),
                )
            encoding_lst.append(p.get_encoding())
            times_lst.append(p.time_secs)
            if print_opt:
                if fun_dict[print_opt](p):
                    print(p.filename)
    return (encoding_lst, times_lst, pvec)

def open_dir(dirpath, **kwargs):
    '''
     `dirpath` is the path of a directory in the specific format (contains a commentary file)
     '''
    print_opt = kwargs.get('print', None)
    pvec = np.zeros(7)
    encoding_lst = []
    times_lst = []
    with open(os.path.join(dirpath, 'latexml_commentary.txt'), 'rb') as comm: 
        log_path = os.path.join(dirpath, 'latexml_errors_mess.txt')
        if os.path.isfile(log_path):
            with open(log_path, 'rb') as log:
                p = ParseLaTeXMLLog(log, comm, os.path.basename(dirpath))
        else:
            log = None
            p = ParseLaTeXMLLog(log, comm, os.path.basename(dirpath))
    assert hasattr(p, 'time_secs'), " Error, %s has no attribute time_secs"%p.filename
    pvec = (fun_dict['success'](p),
        fun_dict['fail'](p),
        fun_dict['fatal'](p),
        fun_dict['maxed'](p),
        fun_dict['timed'](p),
        fun_dict['died'](p),
        fun_dict['notex'](p),
        )
    encoding_lst.append(p.get_encoding())
    times_lst.append(p.time_secs)
    if print_opt:
        if fun_dict[print_opt](p):
            print(p.filename)
    return (encoding_lst, times_lst, pvec)

def summary(summpath, **kwargs):
    '''
    `summpath` is walked finding all article directories
    returns a summary of all the results
    '''
    pvec = np.zeros(7)
    encoding_lst = []
    encoding_tmp = []
    times_lst = []
    for root, dirs, files in os.walk(summpath):
        for tarf in [f for f in files if '.tar' in f]:
            logging.debug('summarizing tarfile: %s'%tarf)
            encoding_tmp, times_tmp, pvec_tmp = open_tar(tarf, **kwargs)
        if 'latexml_commentary.txt' in files:
            logging.debug('summarizing article directory: %s'%dirs)
            encoding_tmp, times_tmp, pvec_tmp = open_dir(root, **kwargs)
        try:
            encoding_lst += encoding_tmp
            times_lst += times_tmp
            pvec += pvec_tmp
        except UnboundLocalError:
            pass
    else: # In the case where summpath is just a single tarfile
        logging.debug('summarizing tarfile: %s'%summpath)
        encoding_lst, times_lst, pvec = open_tar(summpath, **kwargs)

    #for ind, a in enumerate(dir_lst):
    print("Success Fail Fatal Maxed Timed Died no_tex")
    print("{:>7} {:>4} {:>5} {:>5} {:>5} {:>4} {:>6}".format(*list(pvec)))
    print(coll.Counter(encoding_lst))
    cuts = pd.cut(times_lst, 8)
    count = coll.Counter(cuts)
    if np.NAN in count:
        print('NANs', count[np.NAN])
        del count[np.NAN]
    for c in sorted(list(count)):
        print(c, count[c])

if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Stats for documents processed with')
    parser.add_argument('dir_name', type=str,
            help='Path to the processed files')
    parser.add_argument('--print', type=str,
            help='print articles matching this value')
    parser.add_argument("--log", type=str, default=None,
            help='log level, options are: debug, info, warning, error, critical')
    args = parser.parse_args(sys.argv[1:])

    if args.log:
        logging.basicConfig(level=getattr(logging, args.log.upper()))

    summary(args.dir_name, print=args.print)
