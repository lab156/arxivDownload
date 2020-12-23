import networkx as nx
from lxml import etree

def read_pm_tree(fpath):
    '''
    `fpath` is file path to a xml or xml.gz file with the format:
<root>
  <article name="11N05-SternPrime.xml">
    <definition>
      <stmnt> SternPrime _inline_math_ there ...  </stmnt>
      <dfndum>Stern prime</dfndum>
    </definition>
  </article>
  </root>

  Returns: networtx directed graph with nodes the definienda and edges 
  the dependency
    '''
    gxml = etree.parse(fpath).getroot()
    def_dict = {} #keys: dfndum Value: list of hashes of statements where the is appears
        
    for d in gxml.iter(tag = 'dfndum'):
        D = d.getparent().find('stmnt').text
        if d.text.strip() in def_dict:
            def_dict[d.text.strip()].append(hash(D))
        else:
            def_dict[d.text.strip()] = [hash(D),]
    print('Found {} terms in the file.'.format(len(def_dict.values())))

    dgraph = nx.DiGraph()
    empty_str_if_none = lambda s: s if s  else ''
    for k,d_raw in enumerate(def_dict.keys()):
        d = d_raw.strip()
        if k%1000 == 0:
            print('doing k=', k)
        for Def in gxml.iter(tag = 'definition'):
            D = Def.find('.//stmnt')
            #Check if D is not a definition for d
            if hash(D.text) in def_dict[d]:
                pass
            else:
                dfndum_lst = [c.text for c in D.getparent().findall('.//dfndum') ]
                if d in empty_str_if_none(D.text):
                    add_edges_lst = [(d, p.strip()) for p in dfndum_lst if d != p]
                    dgraph.add_edges_from(add_edges_lst)
    return dgraph
