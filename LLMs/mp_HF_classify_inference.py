import logging
import tensorflow as tf
import multiprocessing as mp
import functools 
import queue
import json
import time
import random
import glob
import gzip

#from datetime import datetime
from datetime import datetime as dt
from lxml import etree

from transformers import (AutoTokenizer,
                         TFAutoModelForSequenceClassification,
                         TFBertForSequenceClassification,
                         DataCollatorWithPadding,
                         TFPreTrainedModel,)

import os, sys, inspect
currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
sys.path.insert(0,parentdir+'/embed') 

import peep_tar as peep
import classify_lstm as classy
import parsing_xml as px
from extract import Definiendum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='',
        help="""Path to the trained model 
    ex. /media/hd1/TransformersFineTuned/BertHF1/""")
    parser.add_argument('--out', type=str, default='',
        help="Path to dir 'mine_out_dir' to output mining results.")
    parser.add_argument('--mine', type=str, nargs='+',
            help='''Path to data to mine, ex. /media/hd1/promath/math96
            or  /media/hd1/promath/math96/9601_001.tar.gz''')
    args = parser.parse_args()


    return args

def gen_cfg(args):
    with open(os.path.join(args.model, 'cfg_dict.json'), 'r') as fobj:
         cfg = json.loads(fobj.read())
    cfg['save_path'] = args.out
    cfg['tboard_path'] = os.path.join(args.out, 'tboard_logs') 

    cfg['base_dir'] = os.environ.get('PERMSTORAGE', '/media/hd1') 
    cfg['local_dir'] = os.environ.get('TEMPFASTSTORAGE',
            '/tmp')  # This is temporary fast storage
    #config_path = args.get('configpath', 'config.toml')

    # This is permanent storage# This is permanent storage

    hoy = dt.now()
    timestamp = hoy.strftime("%b-%d_%H-%M")
    #cfg['save_path_dir'] = os.path.join(cfg['local_dir'],
    #        'trans_HF_ner/ner_' + timestamp)
    #os.makedirs(cfg['save_path_dir'], exist_ok=True)
    cfg['min_words'] = 15

    FHandler = logging.FileHandler(cfg['local_dir']+"/training.log")
    logger.addHandler(FHandler)

    return cfg

def get_gpu_info(q="GPU"):
    xla_gpu_lst = tf.config.list_physical_devices(q)
    print( "############################################################")
    print(f"### The len of GPU devices found is: {len(xla_gpu_lst)} ####")
    print(f" {len(xla_gpu_lst)} ")
    print( "############################################################")
    return xla_gpu_lst

def dir_diff(promath_path, inference_path, year_dir=None):
    '''
promath_path ex. '/media/hd1/promath/' 
                 expected file format .tar.gz
inference_path ex. '/media/hd1/glossary/NN.v1/' 
                 expected formar .xml.gz
year_dir: either a string "math91" or a list of strings ["math01", "math02", ]
    '''
    if isinstance(year_dir, str):
        year_dir = [year_dir, ]
        
    # promath and inference path lists
    pp_lst = []
    pi_lst = []
    for year in year_dir:
        pp_lst += [os.path.join(year, os.path.split(n)[-1].split('.')[0])
                for n in glob.glob(os.path.join(promath_path, year) + '/*.tar.gz')]
        pi_lst += [os.path.join(year, os.path.split(n)[-1].split('.')[0])
                for n in glob.glob(os.path.join(inference_path, year) + '/*.xml.gz')]
        # output looks like ['math11/1106_003', '']
        
    pp_set = set(pp_lst)
    pi_set = set(pi_lst)
    return pp_set.difference(pi_set)


def untar_clf_append(tfile, out_path, clf, tzer, cfg, min_words=15):
    '''
    Arguments:
    `tfile` tarfile with arxiv format ex. 1401_001.tar.gz
    `out_path` directory to save the xml.tar.gz file with the same name as `tfile`
    `clf` model with .predict() attribute
    `tzer` hugging face tokenizer object
    '''
    #opt_prob = float(cfg['opt_prob'])
    root = etree.Element('root')
    #print(f"**Peeping into file {fname}  **")
    for fname, tar_fobj in peep.tar_iter(tfile, '.xml'):
        #print(f"**Peeping into file {fname}  **")
        try:
            DD = px.DefinitionsXML(tar_fobj) 
            if DD.det_language() in ['en', None]:
                art_tree = Definiendum(DD, clf, None, None, tzer,\
                        fname=fname, min_words=cfg['min_words']).root
                if art_tree is not None: root.append(art_tree)
        except ValueError as ee:
            print(f"{repr(ee)}, 'file: ', {fname}, ' is empty'")
    return root

def mine_individual_file(_model_, filepath, tzer, cfg, tf_device='/gpu:0'):
    '''
    Mines an individual tar.gz file from promath. More granular mining than `mine_dirs` 
    mines a xml.gz 
    filepath:
        Full path of the the tar.gz file to extract 
             ex. /media/hd1/promath/math01/0103_001.tar.gz

    Saves to a directory with path cfg['save_path']/math01/0103_001.tar.gz

    _model_ may be path of tf.model or a fully usable tf model
    '''
    #opt_prob = float(cfg['opt_prob'])
    logger.info('Classifying the contents of {}'.format(filepath))
    try:
        # data_path is set globally to '' (empty string)
        # example filepath /media/hd1/promath/math01/0103_001.tar.gz
        full_path, tfile = os.path.split(filepath)
        dirname = os.path.split(full_path)[1]
        # expect full_path = /media/hd1/promath/math01
        # dirname = math01
        # tfile = 0103_001.tar.gz
        assert os.path.isdir(full_path), f"{full_path} is not a dir" 
    except FileNotFoundError:
        print(f'{full_path} Not Found')
    out_path = os.path.join(cfg['save_path'], dirname)
    os.makedirs(out_path, exist_ok=True)
   
    #for tfile in tar_lst:
    Now = dt.now()
    #clf = lstm_model
    #vzer = Vectorizer(cfg)

    Model_loaded = _model_   # assume the model is already loaded

    def_root = untar_clf_append(filepath, out_path,\
            Model_loaded, tzer, cfg )
    gz_filename = os.path.basename(tfile).split('.')[0] + '.xml.gz' 
    gz_out_path = os.path.join(out_path, gz_filename) 
    class_time = (dt.now() - Now)

    Now = dt.now()
    with gzip.open(gz_out_path, 'wb') as out_f:
        print("Writing to dfdum zipped file to: %s"%gz_out_path)
        out_f.write(etree.tostring(def_root, encoding='utf8', pretty_print=True))
    writing_time = (dt.now() - Now) 
    logger.info("Writing file to: {} CLASSIFICATION TIME: {} Writing Time {}"\
                     .format(gz_out_path, class_time, writing_time))

def worker_device_lock(name):
    global task_queue, lock, tf_model_dir
    Model = None
    with tf.device(name):
        while not task_queue.empty():
            ind, tf_model_dir, tarfile, tzer, cfg = task_queue.get(timeout=0.5)
            if Model == None:
                lock.acquire()
                Model = TFAutoModelForSequenceClassification.from_pretrained(
                                       os.path.join(tf_model_dir, 'model'))
                #tzer = AutoTokenizer.from_pretrained(
                #        cfg['checkpoint'])

                logger.info(f"Worker {name} has loaded the model {tf_model_dir}.")
                lock.release()
            logger.info(f"Worker {name} is taking file: {tarfile}.")
            mine_individual_file(Model, tarfile, tzer, cfg)
            logger.info(f"Worker {name} finished working on {tarfile}.")
    logger.info(f"Worker {name} is done with the while loop.")

def main_w_permanent_workers():
    """
    Uses the pool-of-permanent-workers model

    singularity run --nv --bind $HOME/Documents/arxivDownload:/opt/arxivDownload,/media/hd1:/opt/data_dir $HOME/singul/runnerN.sif python3 LLMs/mp_HF_classify_inference.py --model /opt/data_dir/TransformersFineTuned/class-2023-06-29_1436 --out /rm_me_path/with_mp_classify --mine /opt/data_dir/promath/math94/940{3,4,5}_001.tar.gz

    """
    args = parse_args()
    print(f'#################Done with args')
    logger = logging.getLogger(__name__)
    cfg = gen_cfg(args)

    # Model directory is a mandatory argument
    global tf_model_dir
    tf_model_dir = args.model


    if args.out != '' :
        mine_out_dir = args.out

    # GET THE PATH AND config

    print(f"{tf_model_dir=}")
    print(f"{os.listdir(os.path.join(tf_model_dir, 'model'))=}")
    #model = TFAutoModelForSequenceClassification.from_pretrained(
    #        os.path.join(tf_model_dir, 'model'))
    #model = TFBertForSequenceClassification.from_pretrained(
    #        os.path.join(tf_model_dir, 'model'))
    print(f'################## Done with model')
    tzer = AutoTokenizer.from_pretrained(
            cfg['checkpoint'])

    # TEST
    #test_result = classy.test_model(model, classy.train_example_path, V, cfg)
    #logger.info(
    #        f'TEST Loss: {test_result[0]:1.3f} and Accuracy: {test_result[1]:1.3f}')

    if args.mine is None:
        logger.info('--mine is empty there will be no mining.')
        raise NotImplementedError('--mine is None, what am I supposed to mine.')

    logger.info('List of Mining dirs: {}'.format(args.mine))
    print('List of Mining dirs: {}'.format(args.mine))

    xla_gpu_lst = get_gpu_info()
    logger.info(f'List of GPUs: {xla_gpu_lst}')

    global lock
    lock = mp.Lock()

    # QUEUE PREPARATION
    global task_queue 
    task_queue = queue.Queue()
    # Fill the index queue
    logger.info(f'Filling task queue:')
    for ind,tarfile in enumerate(args.mine):
        task_queue.put((ind, tf_model_dir, tarfile, tzer, cfg))
        logger.info(f'* Putting {ind} -- {tarfile} ')

    with mp.pool.ThreadPool(len(xla_gpu_lst)) as pool:
        pool.map(worker_device_lock, 
                ['/gpu:'+repr(k) for k in range(len(xla_gpu_lst))])

if __name__ == "__main__":
    #tf.debugging.set_log_device_placement(True)
    main_w_permanent_workers()
