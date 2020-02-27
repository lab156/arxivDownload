import datetime as dt
import re
import dateutil.parser as duparser
import os
import enum
import numpy as np
import collections as coll

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
    NOTEX = enum.auto()  # No TeX file was found it might be pdf only or a weird case like 1806.03429
    FAIL = FATAL | TIMED | MAXED | DIED

class ParseLaTeXMLLog():
    def __init__(self, log_path, max_errors=100):
        '''
        Common actions to do on a latexml errors messages log
        log_path is the path to a latexml_err_mess_ log file
        '''
        if os.path.isfile(log_path):
            self.filename = log_path
            self.dir_name = os.path.split(self.filename)[0]
        else: # log_path is a directory
            self.filename = os.path.join(log_path, 'latexml_errors_mess.txt')
            self.dir_name = log_path

        if os.path.isfile(self.filename):
            with open(self.filename, 'r') as log_fobj:
                self.log = log_fobj.read()

            # Get the time span
            self.start = re.search('\\nprocessing started (.*)\\n',
                    self.log).group(1)
            try:
                self.finish = re.search('\\nprocessing finished (.*)\\n',
                        self.log).group(1)
            except AttributeError:
                self.result = Result.DIED
                self.warnings = self.errors =\
                        self.fatal_errors = self.undefined_macros =\
                        self.missing_files = self.no_prob = np.NAN
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
                self.result = Result.FATAL
                if self.errors > max_errors:
                    self.result |= Result.MAXED
                if self.timedout():
                    self.result |= Result.TIMED

        else:
            assert any(["Main TeX file not found" in line for line in self.commentary()]),\
                    "Error with file %s, don't know what to do in this case"%log_path
            self.result = Result.NOTEX


    def commentary(self):
        '''
        attempts to read the commentary.txt file in a latexml processed directory
        '''
        with open(self.dir_name + '/commentary.txt', 'r') as commentary_fobj:
            comm_str = commentary_fobj.readlines()
        return comm_str

    def get_encoding(self):
        '''
        Tries to get the encoding from the latexml_commentary files
        '''
        find_encod = lambda s: re.search(r'encoding detected: (.*)$', s)
        encod_map = map(find_encod, self.commentary())
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
        result = re.search('Finished in less than (\d+) seconds', self.commentary()[-1])

        if result:
            return int(result.group(1))
        else:
            return None

    def timedout(self):
        '''
        return time if the LAST LINE of the commentary file says it timed out
        return None if process finished on time
        '''
        result = re.search('Timeout of (\d+) seconds occured', self.commentary()[-1])

        if result:
            return int(result.group(1))
        else:
            return None

    def flag(self, max_errors = 100):
        '''
        true if exceeded the number of allowed errors
        '''
        temp_result = None


def summary(dir_lst, **kwargs):
    '''
    args is a list of objects that ParseLaTeXMLLog likes
    returns a summary of all the results
    '''
    pvec = np.zeros(6)

    encoding_lst = []
    for ind, a in enumerate(dir_lst):
        p = ParseLaTeXMLLog(a)
        pvec += (Result.SUCC in p.result,
                p.result in Result.FAIL,
                Result.MAXED in p.result,
                Result.TIMED in p.result,
                Result.DIED in p.result,
                Result.NOTEX == p.result)
        encoding_lst.append(p.get_encoding())
    print("Success Fail Maxed Timed Died no_tex")
    print("{:>7} {:>4} {:>5} {:>5} {:>4} {:>6}".format(*list(pvec)))
    print(coll.Counter(encoding_lst))


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Stats for documents processed with')
    parser.add_argument('dir_name', type=str, nargs='+',
            help='Path to the processed files')
    args = parser.parse_args(sys.argv[1:])
    summary(args.dir_name)


