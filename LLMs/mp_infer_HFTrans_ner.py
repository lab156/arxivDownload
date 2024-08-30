import logging
import tensorflow as tf
import multiprocessing as mp
import os, sys
import functools 
import queue
import time
import random
import glob
from datetime import datetime as dt
from lxml import etree
import json
import gzip
from datasets import Dataset, DatasetDict

from transformers import (AutoTokenizer,
                         TFAutoModelForTokenClassification
A,)

import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import ner.llm_utils as llu
import embed.inference_ner as ninf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def add_dfndum(D, term):
    # Add term to definition element D in a dfndum tag
    dfndum = etree.SubElement(D, 'dfndum')
    dfndum.text = term
    return D

# def tokenizer_mapper(x):
#    tt = tokenizer(x['sentences'], 
#            return_tensors='tf', 
#            is_split_into_words=False, 
#            padding=True, truncation=True)
#    return tt


def prep_raw_data_and_mine(xml_path, tok, tokenizer,  cfg, model):
    '''
    xml_path: path to a compressed xml file of definitions
    
    tok: sentence tokenizer
    
    word_dict: dictionary word -> index
    
    pos_dict: dictionary pos_code -> index
    ---------------------------
    Returns:
    Data ready for the model: word_seq (N,50), pos_seq (N,50), bin_seq (N,50,2)
    def_indices: list with format [(def_tag, [0,1,2,3]), ]
    '''
    if isinstance(xml_path, str):
        if not os.path.isfile(xml_path): raise ValueError(f"{xml_path} is Not a file")
        pars = etree.XMLParser(recover=True)
        xml_tree = etree.parse(xml_path, parser=pars)
        root = xml_tree.getroot()
        Defs = root.findall('.//definition')
    
    def_indices = []
    prev_ind = 0
    all_word_sents = []
    special_token_lst = list(tokenizer.special_tokens_map.values())
    for D in Defs:
        def_parag = D.find('stmnt').text
        word_sents = tok.tokenize(def_parag)
        all_word_sents += word_sents 
        
        # Save D a <definition> and range of individual sentences
        def_indices.append((D, range(prev_ind, prev_ind+len(word_sents))))
        prev_ind += len(word_sents)

    
    # Invert def_indices from (D, range) to
    # inv_indice[k] = D
    inv_indices = len(all_word_sents)*[None]
    for e in def_indices:
        for k in e[1]:
            inv_indices[k] = e[0]
    assert all([not(l == None) for l in inv_indices]), "inv_indices list incomplete"
    
    # INFERENCE AND REGISTRATION

    ibs = cfg['inference_batch_size']
    slice_lst = [slice(s*ibs, (s+1)*ibs, 1) for s in 
            range(len(all_word_sents)//cfg['inference_batch_size'] + 1)]
    if len(all_word_sents)%cfg['inference_batch_size'] == 0:
        slice_lst = slice_lst[:-1]

    if len(all_word_sents) == 0:
        logger.info(
        f'all_word_sents has length zero. {xml_path} has no definitions.')
        print(f'all_word_sents has length zero. {xml_path} has no definitions.')
        return root

    predictions = []
    concat_tokens = []
    for _slice in slice_lst:
        tt = tokenizer(all_word_sents[_slice], return_tensors='tf', 
                is_split_into_words=False, padding=True, truncation=True,
                max_length=cfg['max_length'])

        logits = model(**tt)['logits']
        predicted_ids = tf.math.argmax(logits, axis=-1)

        #print(f"Appending predictions {_slice}")
        predictions += predicted_ids.numpy().tolist()
        concat_tokens += [tt.tokens(j) for j in range(tt['input_ids'].shape[0])]
        # predictions should have shape (n_sentences, n_tokens)

    for i in range(len(all_word_sents)):
        term_lst = llu.crop_terms(concat_tokens[i], 
                [model.config.id2label[p] for p in predictions[i]],
                 golds=all_word_sents[i].split(),
                 special_tokens=special_token_lst)
        for Term in term_lst:
            add_dfndum(inv_indices[i], Term)
    return root 

def mine_individual_file(fname_, sent_tok, tokenizer, cfg, model=None):
    '''
    fname_ is the full path of an .xml.gz file with all the definitions classified 
    eg. /opt/data_dir/glossary/inference_class_all/math96/9601_001.xml.gz
    '''
    print(f"Mining files: {fname_}")
    t3 = dt.now()
    basename = os.path.basename(fname_)
    dirname = os.path.dirname(fname_)
    mathname = os.path.basename(dirname) # ej. math10
    logger.info(f"Files for mining: {fname_}, {basename}") 
    out_root = prep_raw_data_and_mine(fname_,
            sent_tok, tokenizer,
            cfg, model)

    out_dir_math = os.path.join(cfg['outdir'], mathname)
    os.makedirs(out_dir_math, exist_ok=True)
    with gzip.open( os.path.join(out_dir_math, basename), 'wb') as out_fobj:
        out_fobj.write(etree.tostring(out_root, encoding='utf8', pretty_print=True))
    logger.info("Inference time on {}: {}".format(basename, (dt.now() - t3)))

def worker_device_lock(name):
    global task_queue, lock, tf_model_dir
    Model = None
    with tf.device(name):
        while not task_queue.empty():
            ind, xmlfile, sent_tok, cfg = task_queue.get(timeout=0.5)
            if Model == None:
                lock.acquire()
                Model = TFAutoModelForTokenClassification\
                        .from_pretrained(tf_model_dir)
                            
                tokenizer = AutoTokenizer\
                        .from_pretrained(cfg['checkpoint'])
                logger.info(f"Worker {name} has loaded the model {tf_model_dir}.")
                lock.release()
            logger.info(f"Worker {name} is taking file: {xmlfile}.")
            mine_individual_file(xmlfile,
                  sent_tok, tokenizer,
                  cfg, model=Model)
            logger.info(f"Worker {name} finished working on {xmlfile}.")
    logger.info(f"Worker {name} is done with the while loop.")

def get_gpu_info(q="XLA_GPU"):
    xla_gpu_lst = tf.config.list_physical_devices(q)

    print( "############################################################")
    print(f"### The len of GPU devices found is: {len(xla_gpu_lst)} ####")
    print( "############################################################")
    return xla_gpu_lst

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mine', type=str, nargs='+',
            default='/media/hd1/glossary/inference_class_all/',
            help='Path to data to be mined')
    parser.add_argument('--model', type=str,
            default='/home/luis/ner_model',
            help='Path to the tensorflow model directory')
    parser.add_argument('--out', type=str,
            default='/home/luis/NNglossary',
            help='Local path to save')
    parser.add_argument('--senttok', type=str, default=None,
            help='Path to the pickled sentence tokenizer')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    xla_gpu_lst = get_gpu_info('GPU')

    os.makedirs(args.out, exist_ok=True)
    
    FHandler = logging.FileHandler(os.path.join(args.out, 'ner_inference.log'), 
            mode='a')
    logger.addHandler(FHandler)
    logger.info(f'List of XLA GPUs: {xla_gpu_lst}')

    # GET THE PATH AND config
    # Model directory is a mandatory argument
    logger.warning('hola macizo')
    global tf_model_dir
    tf_model_dir = args.model
    cfg = {'outdir': args.out,
            'max_length': 150, 
            'inference_batch_size': 250}
    with open(os.path.join(tf_model_dir, 'config.json')) as fobj:
        cfg['checkpoint'] = json.loads(fobj.read())['_name_or_path']
    logger.info(repr(cfg))

    # GET THE SENTENCE TOKENIZER
    if args.senttok is not None:
        sent_tok = ninf.read_sent_tok(args.senttok)
    else:
        sent_tok = ninf.read_sent_tok(
                os.path.join(tf_model_dir, 'punkt_params.pickle'))
    logger.info(sent_tok._params.abbrev_types)

    if args.mine is None:
        logger.info('--mine is empty there will be no mining.')
        raise NotImplementedError('--mine is None, what am I supposed to mine.')

    logger.info('List of Mining dirs: {}'.format(args.mine))

    global lock
    lock = mp.Lock()

    # QUEUE PREPARATION
    global task_queue 
    task_queue = queue.Queue()
    # Fill the index queue
    logger.info(f'Filling task queue:')
    for ind,xmlfile in enumerate(args.mine):
        task_queue.put((ind, xmlfile,
                  sent_tok, cfg ))
        logger.info(f'* Putting {ind} -- {xmlfile} ')

    with mp.pool.ThreadPool(len(xla_gpu_lst)) as pool:
        pool.map(worker_device_lock, 
                ['/gpu:'+repr(k) for k in range(len(xla_gpu_lst))])

if __name__ == "__main__":
    #tf.debugging.set_log_device_placement(True)
    main()

