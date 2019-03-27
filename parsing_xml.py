import xml.etree.ElementTree as ET
from lxml import etree
import pandas as pd
import re
import sys
import nltk 

#Shortcut to set default to empty string instead Nonetype
empty_if_none = lambda s: s if s else ''



#def recutext1(root, nsstr='{http://dlmf.nist.gov/LaTeXML}'):
#    ret_str = '' #empty_if_none(root.text)
#    for el in list(root):
#        if el.tag != (nsstr + 'Math') and el.tag != (nsstr + 'cite'):
#            ret_str += empty_if_none(el.text)
#            ret_str += empty_if_none(el.tail)
#            ret_str += recutext1(el, nsstr)
#        else:
#            ret_str += empty_if_none(el.tail)
#    return ret_str.lower().replace('\n', ' ')

def text_xml(root, nsstr='{http://dlmf.nist.gov/LaTeXML}'):
    ret_str = empty_if_none(root.text)
    for el in list(root):
        if el.tag == (nsstr + 'Math') :
            if el.attrib.get('mode') == 'inline':
                ret_str += '_inline_math_'
            elif el.attrib.get('mode') == 'display': 
                ret_str += '_display_math_'
            ret_str += empty_if_none(el.tail)
        elif el.tag == (nsstr + 'cite'):
            ret_str += '' # In case you wanna add citation
        else:
            ret_str += empty_if_none(el.text)
            ret_str += empty_if_none(el.tail)
    return ret_str.replace('\n', ' ')

# String to be substituted instead of inline math
math_inline_str = '_inline_math_'
math_display_str = '_display_math_'
cite_str = '_citation_'

xml_dict = { 'math_tag': 'Math',
        # Example: <Math mode="inline" tex="(E,V)" text
        'math_attrib_key': 'mode',
        'math_attrib_display': 'display',
        'math_attrib_inline': 'inline',
        'cite_tag': 'cite',
        }
html_dict = { 'math_tag': 'math',
        'math_attrib_key': 'display',
        'math_attrib_display': 'display',
        'math_attrib_inline': 'inline',
        'cite_tag': 'cite',
        }

def recutext_xml(root, nsstr='{http://dlmf.nist.gov/LaTeXML}'):
    ret_str = empty_if_none(root.text)
    #print('root is', root.tag, root.text, root.tail)
    for el in list(root):
        #print('inside %s tag '%el.tag, el.text, el.tail)
        if el.tag == (nsstr + 'Math'):
            if el.attrib.get('mode') == 'inline':
                ret_str += '_inline_math_'
            elif el.attrib.get('mode') == 'display': 
                ret_str += '_display_math_'
            ret_str += empty_if_none(el.tail)
        elif el.tag == (nsstr + 'cite'):
            ret_str += '_citation_'
            ret_str += empty_if_none(el.tail)
        else:
            #ret_str += empty_if_none(el.text)
            ret_str += recutext_xml(el, nsstr)
            ret_str += empty_if_none(el.tail)
    return ret_str.replace('\n', ' ')

def recutext_html(root, nsstr=''):
    ret_str = empty_if_none(root.text)
    #print('root is', root.tag, root.text, root.tail)
    for el in list(root):
        #print('inside %s tag '%el.tag, el.text, el.tail)
        if el.tag == (nsstr + 'math'):
            if el.attrib.get('display') == 'inline':
                ret_str += '_inline_math_'
            elif el.attrib.get('display') == 'display': 
                ret_str += '_display_math_'
            ret_str += empty_if_none(el.tail)
        elif el.tag == (nsstr + 'cite'):
            ret_str += '_citation_'
            ret_str += empty_if_none(el.tail)
        else:
            #ret_str += empty_if_none(el.text)
            ret_str += recutext_html(el, nsstr)
            ret_str += empty_if_none(el.tail)
    return ret_str.replace('\n', ' ')

class DefinitionsXML(object):
    def __init__(self, file_path):
        '''
        Read an xml file and parse it
        '''
        self.file_path = file_path
        self.filetype = self.file_path.split('.')[-1]
            
        try:
            if self.filetype == 'xml':
                self.exml = etree.parse(file_path, etree.XMLParser(remove_comments=True))
            elif self.filetype == 'html': 
                self.exml = etree.parse(file_path, etree.HTMLParser(remove_comments=True))
        except etree.ParseError:
            raise etree.ParseError('Could not parse file %s'%file_path)

        self.ns = {'latexml': 'http://dlmf.nist.gov/LaTeXML' }
        self.def_lst = []

    def find_definitions(self):
        '''
        finds all definitions in the parsed xml
        and returns a list of them
        '''
        for e in self.exml.iter():
            att_class = e.attrib.get('class', None)
            if att_class:
                if re.match(r'ltx_theorem_[definto]+$', att_class):
                    self.def_lst.append(e)
        return self.def_lst

    def para_p(self, root):
        '''
        The xml format for a definition is as follows:
        <theorem class="ltx_theorem_defn">
        <title>  </title>
        <para>
          <p>
          Here goes the text of the definition
          </p>
          </para>
        </theorem>
        This function gets a theorem root and returns the p tag
        '''
        #return root.findall('./latexml:para/latexml:p', self.ns)[0]
        try:
            root_found = root.findall('.//latexml:para', self.ns)[0]
        except IndexError:
            raise ValueError('para tag not found in file: %s with message: \n %s'
                    %(self.file_path, etree.tostring(root).decode('utf-8')))
        return root_found

    def get_def_text(self):
        '''
        uses the method specified to get the text from 
        the definitions in self.def_lst
        '''
        #Check if there is a list of definitions has been created
        #If not calls the find_definitions method.
        if self.def_lst:
            pass
        else:
            self.find_definitions()
            if self.filetype == 'xml':
                method = recutext_xml
            elif self.filetype == 'html':
                method = recutext_html
            else:
                raise NotImplementedError('method for file type: %s has not been created'%self.filetype)
        return [method(self.para_p(r)) for r in self.def_lst]

    def write_defs(self, path):
        '''
        Append the list of found definitions to the text file at path
        '''
        with open(path,'a') as text_file:
            for d in self.get_def_text():
                text_file.write(d+'\n')
        return 0

    def tag_list(self, tag='p'):
        '''
        Return a list of the specified tag 
        The tags are searched from the self.exml file
        '''
        return self.exml.findall('.//latexml:' + tag, self.ns)



if __name__ == "__main__":
    '''
    Usage: python parsing_xml.py fileList FileToStoreDefs
    '''
    import argparse
    parser = argparse.ArgumentParser(description='parsing xml commandline script')
    parser.add_argument('file_names', type=str, nargs='+',
            help='filenames to find definitions last position is the resulting files')
    parser.add_argument('-l', '--logfile', help='file to write the logs', type=str)
    args = parser.parse_args(sys.argv[1:])


    allfiles = args.file_names[:-1]
    defs_file = args.file_names[-1]
    for f in allfiles:
        try:
            DD = DefinitionsXML(f)
            print('writing file: %s'%f, end='\r')
            DD.write_defs(defs_file)
        except ET.ParseError:
            print('Error parsing file: %s'%f, end='\n')
        # some definitions are empty and have no para tag
        # para_p complains about this and it is important
        # because I don't know the specifics about the format
        except ValueError as e:
            if args.logfile:
                with open(args.logfile, 'a') as log_file:
                    log_file.write(str(e) + '\n')
            else:
                raise e

