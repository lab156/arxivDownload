#!/usr/bin/env python3
# Run with
# mpiexec -n 4 --host 10.0.0.71,10.0.0.72,10.0.0.73,10.0.0.74 python3 /home/pi/arxivDownload/mpi_find_defs.py

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
from math import ceil

comm = MPI.COMM_WORLD
rank = comm.rank
Size = comm.Get_size()
Start = 1991
End = 2020
dir_lst = ['math' + repr(s)[-2:] for s in range(Start, End + 1)]
mnt_path = '/mnt/promath/'


for k,d in enumerate(dir_lst):   # d: math18
    if k%Size == rank:
        try:
            tar_lst = [p for p in  os.listdir(os.path.join(mnt_path, d)) if '.tar.gz' in p]
        except FileNotFoundError:
            print(' %s Not Found'%d)
            break
        print('I am rpi%s and dealing with dir %s \n'%(rank, d))
        out_path = os.path.join('/tmp/', d)
        try:
            os.mkdir(out_path)
        except FileExistsError as ee:
            print(ee, ' continuiung using this directory')
            
        #print(tar_lst)
        #root = etree.Element('root', name=d)
        for tarpath in tar_lst:   # tarpath: 9201_001.tar.gz
            #print(os.path.join(mnt_path, d, T))
            tfile_elm = etree.Element('tarfile', name=tarpath)
            for fname,T in peep.tar_iter(os.path.join(mnt_path, d, tarpath), '.xml'):
                print(fname)
                try:
                    DD = px.DefinitionsXML(T)
                    def_dict = DD.get_def_sample_text_with()
                except ValueError as ee:
                    print("\n Probably empty article: %s \n"%fname, ee) 
                    def_dict = {'real': [], 'nondef': []}
                art_elm = etree.SubElement(tfile_elm, 'article', name=fname)
                for defin in def_dict['real']:
                    defi_elm = etree.SubElement(art_elm, 'definition')
                    defi_elm.text = defin
                for defin in def_dict['nondef']:
                    defi_elm = etree.SubElement(art_elm, 'nondef')
                    defi_elm.text = defin

            #print(etree.tostring(tfile_elm, pretty_print=True).decode('utf-8'))
            gz_filename = os.path.basename(tarpath).split('.')[0] + '.xml.gz' 
            #logging.debug('The name of the gz_filename is: %s'%gz_filename)
            #print('The name of the gz_filename is: %s'%gz_filename)
            gz_out_path = os.path.join(out_path, gz_filename) 
            with gzip.open(gz_out_path, 'wb') as out_f:
                #logging.info("Writing to dfdum zipped file to: %s"%gz_out_path)
                out_f.write(etree.tostring(tfile_elm, pretty_print=True))
