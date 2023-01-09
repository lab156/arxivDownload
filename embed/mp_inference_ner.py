import inference_ner as ninf
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

logger = logging.getLogger(__name__)

def worker_device_lock(name):
    global task_queue, lock, tf_model_dir
    Model = None
    with tf.device(name):
        while not task_queue.empty():
            ind, xmlfile, sent_tok, tkn2idx, pos_ind_dict, cfg = task_queue.get(timeout=0.5)
            if Model == None:
                lock.acquire()
                Model = tf.keras.models.load_model(
                    os.path.join(tf_model_dir, 'bilstm_with_pos'))
                logger.info(f"Worker {name} has loaded the model {tf_model_dir}.")
                lock.release()
            logger.info(f"Worker {name} is taking file: {xmlfile}.")
            ninf.mine_individual_file(xmlfile,
                  sent_tok, tkn2idx,
                  pos_ind_dict,
                  cfg, model=Model)
            logger.info(f"Worker {name} finished working on {xmlfile}.")
    logger.info(f"Worker {name} is done with the while loop.")

def get_gpu_info(q="XLA_GPU"):
    xla_gpu_lst = tf.config.list_physical_devices(q)
    print( "############################################################")
    print(f"### The len of GPU devices found is: {len(xla_gpu_lst)} ####")
    print(f" {len(xla_gpu_lst)} ")
    print( "############################################################")
    return xla_gpu_lst

def main():
    args = ninf.parse_args()

    xla_gpu_lst = get_gpu_info()
    logger.info(f'List of XLA GPUs: {xla_gpu_lst}')

    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    # GET THE PATH AND config
    # Model directory is a mandatory argument
    global tf_model_dir
    tf_model_dir = args.model
    cfg = ninf.open_cfg_dict(os.path.join(tf_model_dir, 'cfg.json'))
    cfg.update({'outdir': args.out})
    logger.info(repr(cfg))
    # GET WORD INDICES
    wind, tkn2idx = ninf.read_word_index_tkn2idx(os.path.join(tf_model_dir,\
            'wordindex.pickle'))
    pos_lst, pos_ind_dict = ninf.read_pos_index_pos2idx(os.path.join(tf_model_dir,\
            'posindex.pickle'))
    logger.info("index of commutative is: {}".format(tkn2idx['commutative']))
    logger.info("POS index of VBD is {}".format(pos_ind_dict['VBD']))
    # GET THE SENTENCE TOKENIZER
    sent_tok = ninf.read_sent_tok(os.path.join(tf_model_dir, 'punkt_params.pickle'))
    logger.info(sent_tok._params.abbrev_types)


    # TEST
    model = tf.keras.models.load_model(
            os.path.join(tf_model_dir, 'bilstm_with_pos'))
    t1 = dt.now()
    ninf.test_model(model, sent_tok, wind, pos_ind_dict, cfg)
    logger.info("Testing time: {}".format((dt.now() - t1)))

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
                  sent_tok, tkn2idx,
                  pos_ind_dict, cfg ))
        logger.info(f'* Putting {ind} -- {xmlfile} ')

    with mp.pool.ThreadPool(len(xla_gpu_lst)) as pool:
        pool.map(worker_device_lock, 
                ['/gpu:'+repr(k) for k in range(len(xla_gpu_lst))])

if __name__ == "__main__":
    tf.debugging.set_log_device_placement(True)
    main()


