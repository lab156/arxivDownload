#!/usr/bin/env python3

from mpi4py import MPI
from random import randint
import time
from itertools import cycle
import parsing_xml as px
from lxml import etree
import gzip
import logging
import os
import peep_tar as peep

comm = MPI.COMM_WORLD
rank = comm.rank
Size = comm.Get_size()
Start = 1991
End = 2020


for q in range(ceil((End - Start)/Size)):
    ###
    year_str = repr(q + rank)
    last2digits = year_str[-2:]
    dirname = math + last2digits
    try:
        tar_lst = os.listdir(dirname)
    except FileNotFoundError:
        break
    print('I am rpi%s and dealing with dir %s \n'%(rank, dirname))
    #for T in tar_lst:
    #    for fname,T in peep.tar_iter(T):
    #        DD = px.DefinitionsXML(T)
    #        DD.get_def_sample_text_with()



