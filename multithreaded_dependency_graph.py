from lxml import etree
from lxml.etree import ElementTree
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import math

import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import itertools

import time  

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_add_edges_lst(i, D, dfndum_lst):
  """
  Input:
   - i: (key,value) pair of def_dict
   - D is the text of a definition
   - dfndum_lst = [c.text for c in D.getparent().findall('.//dfndum') ]

  Output:
    A list of edges to be added to a DGraph object.
  """
  d = i[0]
  v = i[1]
  if hash(D) in v:
    pass
  else:
    add_edges_lst = []
    if d in empty_str_if_none(D):
      for p in dfndum_lst:
        if not d == p:
          add_edges_lst.append((d, p.strip()))
    return add_edges_lst

PLANETMATH_PATH = "TODO"

def do_work(x):
  # time.sleep(0.001)
  return get_add_edges_lst(*x)

def empty_str_if_none(arg):
  if arg:
    return arg
  else:
    return ""

def mk_def_dict(articles):
  """
  Input:
   - an iterable of articles

  Output:
   - def_dict for this tree
  """
  def_dict = {}
  for a in articles:
    for d in a.iter(tag="dfndum"):
      for statement in a.iter(tag="stmnt"):
        D = statement.text
        if d.text.strip() in def_dict:
          def_dict[d.text.strip()].append(hash(D))
        else:
          def_dict[d.text.strip()] = [hash(D),]
  return def_dict

def work(xml_list):
  ag = etree.parse(xml_list[0]).getroot()
  for path in xml_list[1:]:
    ag.append(path)
  return mk_def_dict(list(ag.iter(tag="article")))

def multithreaded_def_dict(xml_list, num_shards):
  xml_lists = list(chunks(xml_list, math.floor(len(xml_list)/num_shards)))
  with ThreadingPool(num_shards) as pool:
    def_dicts = []
    for result in pool.imap_unordered(work, xml_lists):
      def_dicts.append(result)
      
  def_dict = {}

  for new_dict in def_dicts:
    for k,v in new_dict.items():
      def_dict[k] = def_dict.get(k, []) + v

  return def_dict

def multithreaded_dgraph(def_dict, num_threads):
  dgraph = nx.DiGraph()
  def _task_gen():
    for i in def_dict.items():
      for Def in ag.iter(tag="definition"):
        D = Def.find(".//stmnt")
        dfndum_lst = [c.text for c in D.getparent().findall(".//dfndum")]
        yield (i, D.text, dfndum_lst)

  with ThreadPool(num_threads) as pool:
    results = pool.imap_unordered(do_work, _task_gen())
    for result in results:
      if result:
        print("ADDING EDGES FROM: ", result)
        dgraph.add_edges_from(result)
  return dgraph

def _main(xml_files, num_threads=16):
  """
  Input: xml_files - list of full paths to xml files to be processed

  Output: dgraph with all edges added
  """
  def_dict = multithreaded_def_dict(xml_files, num_threads)
  dgraph = multithreaded_dgraph(def_dict, num_threads)

  # print(dgraph) # or whatever you use to print
  print("ok")
  
if __name__ == "__main__":
  pass

  # # stale
  # ag = etree.parse(PLANETMATH_PATH).getroot()
  # articles = list(ag.iter(tag="article"))
  # articles_shards = chunks(articles, 100)
  # def_dicts = [mk_def_dict(x) for x in articles_shards]
  # print(def_dicts)
  # print(len(def_dicts))
  # print(list(len(x) for x in def_dicts))
  # print(sum(list(len(x) for x in def_dicts)))

  # def_dict = {}

  # for new_dict in def_dicts:
  #   for k,v in new_dict.items():
  #     def_dict[k] = def_dict.get(k, []) + v

  # print(len(def_dict))

  # def_dict0 = mk_def_dict(articles)

  # for k, v in def_dict0.items():
  #   assert set(v) == set(def_dict[k])
    
  # # def_dict = {} #keys: dfndum Value: list of hashes of statements where the is appears
  # # # hash_dict = {} # keys: hashes of statements, Values: the text of the statement
  # # # for D in ag.iter(tag = 'stmnt'):
  # # #     hash_dict[hash(D.text)] = D.text

  # # for d in ag.iter(tag = 'dfndum'):
  # #     D = d.getparent().find('stmnt').text
  # #     if d.text.strip() in def_dict: # in case there are repeats
  # #         def_dict[d.text.strip()].append(hash(D))
  # #     else:
  # #         def_dict[d.text.strip()] = [hash(D),]

  # # print("DEF DICT LENGTH: ", len(def_dict))
  # # print("DEF DICT LENGTH: ", len(mk_def_dict(list(ag.iter(tag="article")))))

