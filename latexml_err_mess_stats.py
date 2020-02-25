import datetime as dt
import re
import dateutil.parser as duparser
import os

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

class ParseLaTeXMLLog():
    def __init__(self, log_path):
        '''
        Common actions to do on a latexml errors messages log
        log_path is the path to a latexml_err_mess_ log file
        '''
        with open(log_path, 'r') as log_fobj:
            self.log = log_fobj.read()

        self.filename = log_path

        # Get the conversion stats
        conversion_tuple = parse_conversion(self.log)
        self.warnings = entry_handler(conversion_tuple[1])
        self.errors = entry_handler(conversion_tuple[2])
        self.fatal_errors = entry_handler(conversion_tuple[3])
        self.undefined_macros = entry_handler(conversion_tuple[4])
        self.missing_files = entry_handler(conversion_tuple[5])
        self.no_prob = conversion_tuple[0]


        # Get the time span
        self.start = re.search('\\nprocessing started (.*)\\n',
                self.log).group(1)
        self.finish = re.search('\\nprocessing finished (.*)\\n',
                self.log).group(1)
        d1 = duparser.parse(self.start)
        d2 = duparser.parse(self.finish)
        self.time_secs = (d2-d1).seconds

    def commentary(self):
        '''
        attempts to read the commentary.txt file in a latexml processed directory
        '''
        dir_name = os.path.split(self.filename)[0]
        with open(dir_name + '/commentary.txt', 'r') as commentary_fobj:
            comm_str = commentary_fobj.read()
        return comm_str


