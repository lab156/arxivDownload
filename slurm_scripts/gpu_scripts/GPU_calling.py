import tensorflow as tf
from datetime import datetime as dt
import multiprocessing as mp
import sys

def burn_time(gpu, N = 500000):
    Now = dt.now()
    with tf.device(gpu):
        m1 = tf.constant([[0.0, 1.0],[-1.0, 0.0]])
        m2 = tf.constant([[1.0, 0.0],[0.0, 1.0]])
        for _ in range(N):
            m2 = m2*m1
    burnt_time = (dt.now() - Now)
    print(f"Spent {burnt_time} burning time.")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Test Inference on multiple GPU.')
    parser.add_argument('-N', type=int, default=None,
    help='number of iterations for GPUs e.g. 500000 takes 5 seconds on local system')
    args = parser.parse_args(sys.argv[1:])

    return args
    

def main():
    xla_gpu_lst = tf.config.list_physical_devices("XLA_GPU")
    print( "############################################################")
    print(f"##### The list of GPU devices found is: {xla_gpu_lst} ######")
    print( "############################################################")

    args = parse_args()
    if args.N is not None:
        N_iter = args.N
    else:
        N_iter = 500_000

    for k, gpu in enumerate(xla_gpu_lst):
        Now = dt.now()
        with tf.device(f"/gpu:{k}"):
            ## wont work add gpu burn_time(N_iter)
            pass
        burnt_time = (dt.now() - Now)
        print(f"Spent {burnt_time} burning time.")

def main_parallel():
    xla_gpu_lst = tf.config.list_physical_devices("XLA_GPU")
    print( "############################################################")
    print(f"##### The list of GPU devices found is: {xla_gpu_lst} ######")
    print( "############################################################")

    args = parse_args()
    if args.N is not None:
        N_iter = args.N
    else:
        N_iter = 500_000

    with mp.Pool(processes=len(xla_gpu_lst)) as pool:
        gpu_lst = [(f"/gpu:{k}", N_iter) for k in range(len(xla_gpu_lst))]
        pool.starmap(burn_time, gpu_lst)

    
if __name__ == "__main__":
    main_parallel()
    
