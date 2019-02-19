import requests
from bs4 import BeautifulSoup as bs
import re
import urllib.parse as up

class DefiniendumExtract(object):
    '''
    given a url searchs the webpage for definitions
    and their respective definieda

    Style can be: Wikipedia, GroupProp, Stacks Project
    if None is given the try to guess the style from
    the Domain in the URL
    '''
    def __init__(self, url, style=None):
        self.url = up.urlparse(url)
        self.soup = bs(requests.get(url).text,'lxml')
        if style:
            self.style = style
        else:
            if 'wikipedia' in self.url.hostname:
                self.style = 'wikipedia'
            elif 'subwiki' in self.url.hostname:
                # Subwikis style page
                #Ex. https://groupprops.subwiki.org/wiki/Trivial_group
                self.style = 'subwiki'
            elif 'stacks.math' in self.url.hostname:
                # Subwikis style page
                #Ex. https://stacks.math.columbia.edu/tag/03NB
                self.style = 'stacks_proj'
            else:
                raise NotImplementedError('Could not detect the webpage style')
    def __repr__(self):
        return self.url.geturl()

    def _title_wikipedia(self):
        '''
        This works for the wikipedia and subwiki sites
        '''
        return self.soup.findAll(id = 'firstHeading')[0].text.strip().lower()

    def _title_stacks_proj(self):
        '''
        The tex-section looks like this
        <h2 class="tex-section" id="etale-cohomology-section-presheaves">
              <span data-tag="03NB">54.9</span> Presheaves</h2>

         The strings  method outputs:
             ['54.9', ' Presheaves']
              '''
        return list(self.soup.select('.tex-section')[0].strings)[-1]\
                .strip().lower()

    def title(self):
        option_dict = {'wikipedia': self._title_wikipedia,
         'subwiki': self._title_wikipedia,
         'stacks_proj': self._title_stacks_proj,}
        return option_dict[self.style]()

    def _defin_section_wikipedia(self):
        reg_expr = re.compile('.*definition', re.I)
        try:
            ret_search = self.soup.findAll('span', id= reg_expr)[0]
        except IndexError:
            ret_search = None
        return ret_search

    def _defin_section_stacks_proj(self):
        return self.soup.select('.env-definition')[0]

    def defin_section(self):
        '''
        There tends to be a definition section in all the syles
        of webpages. This function tries to find this section
        and returns the BeautifulSoup element of the definition
        '''
        option_dict = {'wikipedia': self._defin_section_wikipedia,
         'subwiki': self._defin_section_wikipedia,
         'stacks_proj': self._defin_section_stacks_proj,}
        return option_dict[self.style]()

    def _next_wikipedia(self, num_links=1):
        '''
        Returns a list of url that might followable
        when scraping each website
        '''
        link_lst = []
        # get the next link
        # Ex. /wiki/Differentiable_manifold
        temp_section = self.defin_section().findNext('p')
        while  len(link_lst) <= num_links:
            try:
                temp_lst = temp_section\
                        .findChildren('a')
                link_lst += [up.urljoin(self.url.geturl(), L['href'])\
                        for L in temp_lst]
            except AttributeError:
                pass
            try:
                # If no more paragraphs can be found
                # exit gracefully with what we have
                temp_section = temp_section.findNext('p')
            except AttributeError:
                break
        return link_lst

#TODO: Determine if this method is better than the one being used
# in the _next_wikipedia function
    def _next_subwiki(self, num_links=1):
        def search_assist(x):
            if x:
                has_wiki = ('wiki' in x)
                no_colon = not (':' in  x)
                return has_wiki and  no_colon
            else:
                #there are <a> tags with no reference
                return False
        temp_lst = self.soup.findAll('a', href=search_assist)
        return [up.urljoin(self.url.geturl(), L['href'])\
                for L in temp_lst ][: num_links]

    def _next_stacks_proj(self):
        '''
        Returns a list of url that might followable
        when scraping each website
        '''
        link_lst = []
        # get the next navigation tag
        # Ex. tag/03NF
        next_link = self.soup.select('.right')[0].findChild('a')['href']
        # join to the main website domain
        link_lst.append(up.urljoin(
            self.url.geturl(),
            next_link))
        return link_lst

    def next(self, **kwargs):
        option_dict = {'wikipedia': self._next_wikipedia,
         'subwiki': self._next_subwiki,
         'stacks_proj': self._next_stacks_proj,}
        links_lst = option_dict[self.style](**kwargs)
        return links_lst

    def def_pair_or_none(self):
        definition_section = self.defin_section()
        if definition_section:
            tmp_paragraph = definition_section.findNext('p').text 
        else:
            # Some article don't have a Definition Section
            # We have to survive that
            return None
        if self.title().lower() in tmp_paragraph.lower():
            return self.title(), tmp_paragraph
        else:
            return None
