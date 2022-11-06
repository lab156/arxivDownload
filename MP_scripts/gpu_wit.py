import tensorflow as tf
import multiprocessing as mp

def mat_mul(gpu):
    with tf.device(gpu):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)

def para_exec():
    gpu_lst = [f"/gpu:{k}" for k in range(2)]
    with mp.Pool(1) as pool:
        print(pool.map(mat_mul, gpu_lst))
        
if __name__ == "__main__":
    gpu_lst = [f"/gpu:{k}" for k in range(2)]
    with mp.Pool(1) as pool:
        print(pool.map(mat_mul, gpu_lst))