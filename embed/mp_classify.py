import classify_lstm as classy
import logging
import tensorflow as tf
import multiprocessing as mp
import os, sys
import functools 
import queue
import time
import random

logger = logging.getLogger(__name__)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='',
        help="Path to the trained model ex. '/media/hd1/trained_models/lstm_classifier/lstm_Aug-19_04-15")
    parser.add_argument('--out', type=str, default='',
        help="Path to dir 'mine_out_dir' to output mining results.")
    parser.add_argument('--mine', type=str, nargs='+',
            help='Path to data to mine, ex. /media/hd1/promath/math96')
    args = parser.parse_args()

    return args

def worker_device(name):
    global task_queue
    #time.sleep(5*random.random()) # wait some random time
    sleep_secs = int(name[-1])
    logger.info(f"Device {name} is sleeping for {sleep_secs} seconds.")
    time.sleep(sleep_secs)
    while not task_queue.empty():
        with tf.device(name):
            ind, tf_model_dir, tarfile, V, cfg = task_queue.get(timeout=0.5)
            logger.info(f"Device {name} is taking file: {tarfile}.")
            classy.mine_individual_file(tf_model_dir, tarfile, V, cfg)

def get_gpu_info(q="XLA_GPU"):
    xla_gpu_lst = tf.config.list_physical_devices(q)
    print( "############################################################")
    print(f"### The len of GPU devices found is: {len(xla_gpu_lst)} ####")
    print(f" {len(xla_gpu_lst)} ")
    print( "############################################################")
    return xla_gpu_lst

def main_w_permanent_workers():
    """
    Uses the pool-of-permanent-workers model
    """
    args = parse_args()
    logger = logging.getLogger(__name__)

    # Model directory is a mandatory argument
    tf_model_dir = args.model

    if args.out != '' :
        mine_out_dir = args.out

    # GET THE PATH AND config
    cfg = classy.open_cfg_dict(os.path.join(tf_model_dir, 'cfg_dict.json'))
    cfg['save_path'] = mine_out_dir
    cfg['tboard_path'] = os.path.join(mine_out_dir, 'tboard_logs') 
    V = classy.Vectorizer(os.path.join(tf_model_dir,'idx2tkn.pickle'), cfg)
    print('Index of commutative is: ', V.tkn2idx['commutative'])

    model = classy.load_model_logic(cfg, tf_model_dir)

    # TEST
    test_result = classy.test_model(model, classy.train_example_path, V, cfg)
    logger.info(
            f'TEST Loss: {test_result[0]:1.3f} and Accuracy: {test_result[1]:1.3f}')

    if args.mine is None:
        logger.info('--mine is empty there will be no mining.')
        raise NotImplementedError('--mine is None, what am I supposed to mine.')

    logger.info('List of Mining dirs: {}'.format(args.mine))

    xla_gpu_lst = get_gpu_info()
    logger.info(f'List of XLA GPUs: {xla_gpu_lst}')

    # QUEUE PREPARATION
    global task_queue 
    task_queue = queue.Queue()
    # Fill the index queue
    logger.info(f'Filling task queue:')
    for ind,tarfile in enumerate(args.mine):
        task_queue.put((ind, tf_model_dir, tarfile, V, cfg))
        logger.info(f'* Putting {ind} -- {tarfile} ')

    with mp.pool.ThreadPool(len(xla_gpu_lst)) as pool:
        pool.map(worker_device, ['/gpu:'+repr(k) for k in range(len(xla_gpu_lst))])

def main():
    '''
    Usage example:
    singularity run --nv --bind $HOME/Documents/arxivDownload:/opt/arxivDownload,/media/hd1:/opt/data_dir $HOME/singul/runner.sif python3 embed/mp_classify.py --model /opt/data_dir/trained_models/lstm_classifier/lstm_Aug-19_17-22 --out /rm_me_path/with_mp_classify --mine /opt/data_dir/promath/math94/940{3,4,5}_001.tar.gz
    '''
    #mp.set_start_method('spawn', force=True)
    args = parse_args()

    # Model directory is a mandatory argument
    tf_model_dir = args.model

    if args.out != '' :
        mine_out_dir = args.out

    # GET THE PATH AND config
    cfg = classy.open_cfg_dict(os.path.join(tf_model_dir, 'cfg_dict.json'))
    cfg['save_path'] = mine_out_dir
    cfg['tboard_path'] = os.path.join(mine_out_dir, 'tboard_logs') 
    V = classy.Vectorizer(os.path.join(tf_model_dir,'idx2tkn.pickle'), cfg)
    print('Index of commutative is: ', V.tkn2idx['commutative'])

    model = classy.load_model_logic(cfg, tf_model_dir)

    # TEST
    test_result = classy.test_model(model, classy.train_example_path, V, cfg)
    logger.info(
            f'TEST Loss: {test_result[0]:1.3f} and Accuracy: {test_result[1]:1.3f}')

    if args.mine is not None:
        logger.info('List of Mining dirs: {}'.format(args.mine))

        xla_gpu_lst = tf.config.list_physical_devices("XLA_GPU")
        logger.info(f'List of XLA GPUs: {xla_gpu_lst}')

        #with mp.Pool(processes=len(xla_gpu_lst)) as pool:
        with mp.pool.ThreadPool(processes=len(xla_gpu_lst)) as pool:
            tarfile_lst = [(tf_model_dir, f, V, cfg) for f in args.mine]
            pool.starmap(classy.mine_individual_file, tarfile_lst)
    else:
        logger.info('--mine is empty there will be no mining.')

if __name__ == "__main__":
    tf.debugging.set_log_device_placement(True)
    main_w_permanent_workers()
