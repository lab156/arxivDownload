import xml.etree.ElementTree as ET
from lxml import etree
import pandas as pd
import re
import sys
import nltk
import random
import os.path

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
        else:
            #ret_str += empty_if_none(el.text)
            ret_str += recutext_html(el, nsstr)
            ret_str += empty_if_none(el.tail)
    return re.sub('\s+', ' ', ret_str.replace('\n', ' '))

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
                self.recutext = recutext_xml
            elif self.filetype == 'html': 
                self.exml = etree.parse(file_path, etree.HTMLParser(remove_comments=True))
                self.recutext = recutext_html
        except etree.ParseError as e:
            print('The file ', file_path, ' produced an error: ', e)
            raise ValueError('XML Syntax error')
        except etree.XMLSyntaxError as e:
            print('The file ', file_path, ' produced an error: ', e)

        self.ns = {'latexml': 'http://dlmf.nist.gov/LaTeXML' }
        self.def_lst = []

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

    def create_xml_branch(self):
        '''
        Sat 28 Sep 2019
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

    def get_definiendum(self):
        pass

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

    def get_def_sample_text_with(self, sample_size=3, min_words=15):
        '''
        Starts by running `get_def_text` and then find a random sample of nondefinitions
        because it checks that the paragraphs is already contained in the definitions

        Returns a dictionary with `realdef` list of definitions
        and a `nondef` list of paragraphs text
        '''
        text_dict = {'real': self.get_def_text()}
        para_lst_nonrand = self.para_list()

        para_lst = random.sample(para_lst_nonrand, sample_size)
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
                print('article %s has messed up para'%f)
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
    parser.add_argument('-l', '--logfile', help='file to write the logs', type=str)
    args = parser.parse_args(sys.argv[1:])


    allfiles = args.file_names[:-1]
    defs_file = args.file_names[-1]
    for f in allfiles:
        try:
            DD = DefinitionsXML(f)
            print('writing file: %s'%f, end='\r')
            DD.write_defs(defs_file)
        except TypeError:
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

