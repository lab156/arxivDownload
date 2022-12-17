import tensorflow as tf
import multiprocessing as mp
from time import sleep

def mat_mul(gpu):
    # multiply two matrices on the `gpu` device
    with tf.device(gpu):
        print(f'mat multi on {gpu}')
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
    sleep(5)
    
def long_mat_mul(gpu):
    # multiply two matrices on the `gpu` device
    with tf.device(gpu):
        print(f'mat multi on {gpu}')
        for _ in range(10_000):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)

def para_exec():
    gpu_lst = [f"/gpu:{k}" for k in range(2)]
    with mp.Pool(2) as pool:
        print(pool.map(mat_mul, gpu_lst))
        
def paral_threadpool_exec():
    gpu_lst = [f"/gpu:{k}" for k in [0,1,0,1]]
    with mp.pool.ThreadPool(2) as pool:
        print(pool.map(long_mat_mul, gpu_lst))
        
def main_pool_gpu_lst():
    gpu_lst = [f"/gpu:{k}" for k in range(1)]
    with mp.Pool(1) as pool:
        print(pool.map(mat_mul, gpu_lst))
        
def main_threadpool_gpu_lst():
    gpu_lst = [f"/gpu:{k}" for k in range(2)]
    with mp.ThreadPool(2) as pool:
        print(pool.map(mat_mul, gpu_lst))
        
if __name__ == "__main__":
    paral_threadpool_exec()
