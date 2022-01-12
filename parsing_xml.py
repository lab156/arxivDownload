# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
from lxml import etree
import pandas as pd
import re
import sys
import nltk
import random
import os.path
from collections import defaultdict, Counter
#import magic
import tarfile
import peep_tar as peep
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from enum import Flag, auto
from io import StringIO
import gzip


#Shortcut to set default to empty string instead Nonetype
empty_if_none = lambda s: s if s else ''

def check_sanity(p, ns):
    '''
    Input:
    p: element tree result of searching for para tags
    Checks:
    contains an ERROR tag
    '''
    if p.findall('.//latexml:ERROR', ns):
        return False
    else:
        return True

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
        'item_tag': 'item',
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
        try:
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
            elif el.tag == (nsstr + 'ERROR'):
                #Ignore error messages such as:
                #<ERROR class="undefined">\pmowner</ERROR>djao24
                pass
            elif el.tag == (nsstr + 'tags') and el.getparent().tag == (nsstr + 'item'):   
                #Catch this pattern:
                #<item>
                #  <tags>
                ret_str += '_item_'
                ret_str += empty_if_none(el.tail)
            elif el.tag == (nsstr + 'tag') and (el.attrib.get('role') == 'refnum' or el.attrib.get('role') == 'typerefnum'):
                #Tags have a `role` attrib with text and this should be ignored
              #<tags>
              #  <tag>2.</tag>
              #  <tag role="refnum">2</tag>
              #  <tag role="typerefnum">item 2</tag>
              #</tags>
                pass
            else:
                #ret_str += empty_if_none(el.text)
                if el.tag == (nsstr + 'break'):
                    # Substitute the line break for a blank space
                    ret_str += ' '
                ret_str += recutext_xml(el, nsstr)
                ret_str += empty_if_none(el.tail)
        except AttributeError as EE: 
            print(EE,'\n')
            import pdb; pdb.set_trace()
    # remove any multiple space and trailing whitespaces
    return re.sub('\s+', ' ', ret_str.replace('\n', ' '))

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
        elif el.tag == (nsstr + 'table'):
            if el.attrib.get('class') == "ltx_equation ltx_eqn_table":
                ret_str += '_display_math_'
            ret_str += empty_if_none(el.tail)
        elif el.tag == (nsstr + 'cite'):
            ret_str += '_citation_'
            ret_str += empty_if_none(el.tail)
        elif el.tag == (nsstr + 'span') and el.getparent().attrib.get('class') == 'ltx_item':
            ret_str += '_item_'
            ret_str += empty_if_none(el.tail)
        else:
            #ret_str += empty_if_none(el.text)
            ret_str += recutext_html(el, nsstr)
            ret_str += empty_if_none(el.tail)
    return re.sub('\s+', ' ', ret_str.replace('\n', ' '))

class EmptyXMLError(Exception):
    def __init__(self, fname=None):
        if fname is not None:
            self.fname = fname
        else:
            self.fname = '<Unknown File Name>'
        self.message = '{} is empty.'.format(self.fname)
        super().__init__(self.message)

class ParsingResult(Flag):
    SUCC = auto()
    EMPTY = auto()
    FAIL = auto()
        
class DefinitionsXML(object):
    def __init__(self, file_path, fname=None):
        '''
        Read an xml file and parse it
        '''
        self.BAD_BASESTRING_CHARS = None
        empty_xml = etree.fromstring('<document> Empty file </document>')
        if isinstance(file_path, str):
            self.file_path = file_path
            self.filetype = self.file_path.split('.')[-1]
        elif hasattr(file_path, 'read'):
            # FileObj: assume that it is xml
            if fname is not None:
                self.file_path = fname
            else:
                # In extracted file object the name of the tar file is the best
                # we can do in the case when the fname is not provided
                self.file_path = file_path.name
            self.filetype = 'xml'
        else:
            raise NotImplementedError('file_path is not str nor has read() attrib')

        try:
            if self.filetype == 'xml':
                self.exml = etree.parse(file_path, etree.XMLParser(remove_comments=True))
                self.recutext = recutext_xml
                self.parse = ParsingResult.SUCC
            elif self.filetype == 'html': 
                self.exml = etree.parse(file_path, etree.HTMLParser(remove_comments=True))
                self.recutext = recutext_html
                self.parse = ParsingResult.SUCC
            else:
                raise NotImplementedError("Filetype: %s not implemented yet"%\
                        self.filetype)
        except etree.XMLSyntaxError as e:
            if 'Document is empty' in e.args[0]:
                #raise EmptyXMLError(self.file_path)
                print('The file ', self.file_path, ' is empty.')
                self.exml = empty_xml 
                self.parse = ParsingResult.EMPTY
            elif  'invalid character in attribute value' in e.args[0]:
                if self.filetype == 'xml': 
                    self.fix_bad_chars(file_path)
                    self.recutext = recutext_xml
                    self.parse = ParsingResult.SUCC
                    print('The file ', self.file_path, ' recovered from an error: ', e)
                else:
                    raise NotImplementedError('recover not implemented for html')
            else:
                print('XML ParseError -- {} -- {}'.format(self.file_path, e))
                self.exml = empty_xml 
                self.parse = ParsingResult.FAIL

        self.ns = {'latexml': 'http://dlmf.nist.gov/LaTeXML' }
        self.def_lst = []

    def fix_bad_chars(self, file_path):
        BAD = []
        for i in range(0, 10000):
            try:
                x = etree.parse(StringIO('<p>%s</p>' % chr(i)))
            except etree.XMLSyntaxError:
                BAD.append(i)
        # 38 and 60 are start tag and & 
        self.BAD_BASESTRING_CHARS = [chr(b) for b in BAD if b not in [38, 60]]

        #with open(file_path, 'rb') as fobj:
        file_path.seek(0,0)
        file_str = file_path.read().decode('utf8')
        print(file_str[:100])
        #with open("/home/luis/rm_me_work_on_px/out.xml", 'w') as outfobj:
        #    outfobj.write(file_str)
        for b in self.BAD_BASESTRING_CHARS:
            file_str = file_str.replace(b, '')

        if self.filetype == 'xml':
            self.exml = etree.XML(file_str.encode('utf8'))
        else:
            raise NotImplementedError('no HTML implementation at fix_bad_chars')

    def find_definitions(self):
        '''
        finds all definitions in the parsed xml
        and returns a list of them
        '''
        if self.filetype == 'xml':
            for e in self.exml.iter():
                att_class = e.attrib.get('class', None)
                if att_class:
                    if re.match(r'ltx_theorem_[definto]+$', att_class):
                        self.def_lst.append(e)
        elif self.filetype == 'html':
            self.def_lst = self.exml.xpath(".//div[re:match(@class, 'ltx_theorem_[definto]+$')]",
                    namespaces={"re": "http://exslt.org/regular-expressions"})
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
            if self.filetype == 'xml':
                root_found = root.findall('.//latexml:para', self.ns)[0]
            elif self.filetype == 'html':
                root_found = root.xpath(".//div[contains(@class, 'ltx_para')]" )[0]
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

        return [self.recutext(self.para_p(r)) for r in self.def_lst]

    def det_language(self, sample_size=5, min_words=15, start_at=5):
        """
        Use langdetect library to find the language of the article.
        """
        lang_counter = Counter()
        for k, p in enumerate(self.para_list()):
            p_text = self.recutext(p)
            if len(p_text.split()) >= min_words and k > start_at:
                try:
                    lang_counter.update([detect(p_text)])
                except LangDetectException:
                    print(f'Article {self.file_path} gave no features error')
            if sum(lang_counter.values()) >= sample_size:
                break
        else:
            # if the whole para_list was looped without finding good paragraphs
            return None
        #assert len(set(lang_list)) == 1, f"""Article {self.file_path} has multiple
        #        languages {lang_list}"""
        return lang_counter.most_common()[0][0]


    def get_def_sample_text_with(self, sample_size=3, min_words=15):
        '''
        Starts by running `get_def_text` and then find a random sample of nondefinitions
        because it checks that the paragraphs is already contained in the definitions

        Returns a dictionary with `realdef` list of definitions
        and a `nondef` list of paragraphs text
        '''
        text_dict = {'real': self.get_def_text()}
        para_lst_nonrand = self.para_list()

        try:
            para_lst = random.sample(para_lst_nonrand, sample_size)
        except ValueError as ee:
            para_lst = []
            print('article %s does not have enough paragraphs to \
                    sample'%self.file_path)
        return_lst = []
        #create list of para inside def tags to check for repeats
        check_repeats = set(map(self.para_p, self.def_lst))
        for p in para_lst:
            # make sure that the p tag is not in the def list
            if check_sanity(p, self.ns):
                para_text =  self.recutext(p)
                # make sure that the p tag is not in the def list and check min_words
                if len(para_text.split()) >= min_words and p not in check_repeats:
                    return_lst.append(para_text)
            else:
                print('article %s has paragraphs with reported errors'%self.file_path)
        text_dict['nondef'] = return_lst
        return text_dict

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

    def para_list(self):
        '''
        returns a list of the para tags 
        '''
        if self.filetype == 'xml':
            return self.exml.findall('.//latexml:para', self.ns)
        elif self.filetype == 'html':
            return self.exml.xpath(".//div[contains(@class, 'ltx_para')]")

    def run_recutext_onall_para(self, cleaner_fun=None, joiner_fun=None):
        '''
        run recutext on all para tags in self.
        Outputs xml with format:
        <article name=...>
            <parag ind=INT>
            text text
            </parag>
        </article>
        If cleaner_fun is given, this function is run on the text.
        '''
        article_elem = etree.Element('article')
        article_elem.attrib['name'] = self.file_path

        if self.parse == ParsingResult.SUCC:
            for ind, par in enumerate(self.para_list()):
                text = self.recutext(par)
                if cleaner_fun is not None:
                    text = cleaner_fun(text)
                if joiner_fun is not None:
                    text = joiner_fun(text)

                parag = etree.Element("parag")
                parag.attrib['index'] = repr(ind) 
                parag.text = text
                article_elem.append(parag)
        elif self.parse == ParsingResult.EMPTY:
            article_elem.attrib['empty'] = 'true'
        return article_elem


# Inherit from DefinitionsXML
#
class StacksProjectXML(DefinitionsXML):
    def __init__(self, file_path):
        super().__init__(file_path)

    def create_xml_branch(self):
        '''
        This is intended to work only to the stacks project xml at first
        self being one xml file create a branch with the following structure

        <article name="chow.xml">
            <definition>
              <stmnt> Let _inline_math_ be a ring.  complex over _inline_math_ </stmnt>
              <dfndum>-periodic complex</dfndum>
              <dfndum>cohomology modules</dfndum>
              <dfndum>exact</dfndum>
            </definition>
      '''
        branch = etree.Element('article')
        branch.attrib['name'] = os.path.basename(self.file_path)
        # for each tag like <theorem class="ltx_theorem_definition" in def_lst
        for defi_rt in self.find_definitions():
            root = etree.Element("definition")
            #root.attrib['index'] = repr(ind) -- not defining an index
            statement = etree.SubElement(root, 'stmnt')
            statement.text = self.recutext(self.para_p(defi_rt))
            all_dfndum = defi_rt.xpath('.//latexml:text[contains(@font, "italic")]',
                    namespaces=self.ns)
            for dum in all_dfndum:
                dfndum = etree.SubElement(root, 'dfndum')
                dfndum.text = dum.text
            branch.append(root)
        return branch


class PlanetMathXML(DefinitionsXML):
    def __init__(self, file_path, also_see_tex=True):
        '''
        If the original .tex files are sitting beside the .xml files 
        and the `also_see_tex` flag is set to True then we search the 
        the file of the commands of the form \pmmeta
        
        If the XML output has error tags we can get the meta data by:
        add support for the planet math pmmeta headers ex.
        <ERROR class="undefined">\pmtype</ERROR>Definition
        <ERROR class="undefined">\pmcomment</ERROR>trigger rebuild
        '''
        super().__init__(file_path)
        if also_see_tex:
            # Search for the .tex file with the same path just different filetype
            with open(file_path[:-4] + '.tex', 'r') as tex_fobj:
                self.tag_lst = re.findall('\\\\(pm[a-z]+)\{(.*?)\}', tex_fobj.read())
                # This has the format: [('pmtype', 'Definition'), ('pmcomment', 'trigger rebuild')]
            self.tag_vals = defaultdict(list)
            for t in self.tag_lst:
                if t[1]:
                    self.tag_vals[t[0]].append(t[1])
                else:
                    # just call to create an entry
                    self.tag_vals[t[0]]

        else:
            self.tag_lst = self.exml.xpath('.//latexml:ERROR', namespaces=self.ns)
            self.tag_vals = defaultdict(list)
            for t in self.tag_lst:
                # Ex. t.text: '\\pmtype' and  t.tail: 'Definition\n' 
                if t.text.startswith('\\pm'):
                    trim_str = t.text[1:]
                    if t.tail:
                        self.tag_vals[trim_str].append( t.tail.strip() )

    def extract_text(self):
        return_str = ""
        for p in self.para_list():
            return_str += self.recutext(p)
        return return_str

    def create_xml_branch(self):
        branch = etree.Element('article')
        branch.attrib['name'] = os.path.basename(self.file_path)

        root = etree.Element("definition")
        statement = etree.SubElement(root, 'stmnt')
        def_text = self.extract_text()
        statement.text = def_text

    # sometimes tag_vals do not contain a pmdefines value but the title seems usable
    #example: 26B05-LogarithmicDerivative.xml 'pmdefines': [] 'pmtitle': ['logarithmic derivative']

        if self.tag_vals['pmdefines'] != []:
            dum = self.tag_vals['pmdefines'][0]
        elif self.tag_vals['pmtitle'] != []:
            dum = self.tag_vals['pmtitle'][0]
        else:
            # make sure the next condition is never true
            dum = 'never go in there'
            def_text = ''

        if dum.lower() in def_text.lower():
            dfndum = etree.SubElement(root, 'dfndum')
            dfndum.text = dum

        branch.append(root)
        return branch


if __name__ == "__main__":
    '''
    Usage: python parsing_xml.py fileList FileToStoreDefs
    This command finished processing all math.AG files of 2015

     python parsing_xml.py ~/media_home/math.AG/2015/*/*.xml ../new_real_defs.txt -l ../errors_new_real_defs.txt

     Observe that the -l flag is necessary for it to finish
    '''
    import argparse
    parser = argparse.ArgumentParser(description='parsing xml commandline script')
    parser.add_argument('file_names', type=str, nargs='+',
            help='filenames to find definitions last position is the resulting files')
    parser.add_argument('--defs', help='file to write the the real definitions', type=str)
    parser.add_argument('--nondefs', help='file to write the the non-definitions', type=str)
    parser.add_argument('-l', '--logfile', help='file to write the logs', type=str)
    args = parser.parse_args(sys.argv[1:])


    allfiles = args.file_names[:-1]
    defs_file = args.file_names[-1]
    for f in allfiles:
        if True:   #magic.detect_from_filename(f).mime_type == 'application/gzip':
            raise ValueError('This is where I need magic')
            try:
                for fobj in peep.tar_iter(f, '.xml'):
                    print('hola')
                    DD = DefinitionsXML(fobj)
                    print('writing file: %s'%f, end='\r')
                    DD.write_defs(defs_file)
            except TypeError as ee:
                print('Error parsing file: %s ::: %s'%(f, ee), end='\n')
            # some definitions are empty and have no para tag
            # para_p complains about this and it is important
            # because I don't know the specifics about the format
            except ValueError as e:
                if args.logfile:
                    with open(args.logfile, 'a') as log_file:
                        log_file.write(str(e) + '\n')
                else:
                    raise e

