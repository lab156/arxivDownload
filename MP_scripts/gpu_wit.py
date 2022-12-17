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
        for _ in range(100):
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
        
def main_create_2_gpus():   
    gpus = tf.config.list_physical_devices('GPU')
    print(f'#######  Found gpu-s {gpus} ##########')
    # this is to print GPU debugging info
    tf.debugging.set_log_device_placement(True)
    if gpus:
      # Create 2 virtual GPUs with 1GB memory each
      try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
             tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print("###### ", len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
        
    paral_threadpool_exec()
    
        
if __name__ == "__main__":
    #paral_threadpool_exec()
    main_create_2_gpus()
        
        
        
        
        
        
        
        
        
        
