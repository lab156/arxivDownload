import classify_lstm as classy
import logging
import tensorflow as tf
import multiprocessing as mp
import os
from functools import partial

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

def tup_print(uno, dos, tres):
    print(uno, dos, tres)

def main():
    mp.set_start_method('spawn', force=True)
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
    print(V.tkn2idx['commutative'])

    model = classy.load_model_logic(cfg, tf_model_dir)

    # TEST
    test_result = classy.test_model(model, classy.train_example_path, V, cfg)
    logger.info(
            f'TEST Loss: {test_result[0]:1.3f} and Accuracy: {test_result[1]:1.3f}')

    if args.mine is not None:
        logger.info('List of Mining dirs: {}'.format(args.mine))

        xla_gpu_lst = tf.config.list_physical_devices("XLA_GPU")
        logger.info(f'List of XLA GPUs: {xla_gpu_lst}')

        with mp.Pool(processes=len(xla_gpu_lst)) as pool:
            tarfile_lst = [(tf_model_dir, f, V, cfg) for f in args.mine]
            #pool.starmap(tup_print, tarfile_lst)
            pool.starmap(classy.mine_individual_file, tarfile_lst)
    else:
        logger.info('--mine is empty there will be no mining.')

if __name__ == "__main__":
    main()
