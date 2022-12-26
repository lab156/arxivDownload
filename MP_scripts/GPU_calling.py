import tensorflow as tf
from datetime import datetime as dt
import multiprocessing as mp
import sys
import time
import queue
import threading
import random
import functools

def burn_time(gpu, N = 50000):
    Now = dt.now()
    with tf.device(gpu):
        m1 = tf.constant([[0.0, 1.0],[-1.0, 0.0]])
        m2 = tf.constant([[1.0, 0.0],[0.0, 1.0]])
        for _ in range(N):
            m2 = m2*m1
    burnt_time = (dt.now() - Now)
    print(f"Spent {burnt_time} burning time.")


def wait_time(gpu, N = 3, ind=-1):
    #global task_queue, counter
    print(f"GPU: {gpu} -- sleeping for {N} secs on task {ind}", flush=True)
    time.sleep(N)
    #counter.release()
    return gpu

def task_fun(N):
    global task_queue, counter
    return functools.partial(wait_time, N=N) 


def callback(arg):
    global counter
    # decrement the counter
    print('callback was called')
    counter.acquire()

def get_gpu_info(q="XLA_GPU"):
    xla_gpu_lst = tf.config.list_physical_devices(q)
    print( "############################################################")
    print(f"### The len of GPU devices found is: {len(xla_gpu_lst)} ####")
    print(f" {len(xla_gpu_lst)} ")
    print( "############################################################")
    return xla_gpu_lst

def main_w_queue_semafore():
    # Based on https://superfastpython.com/parallel-nested-for-loops-in-python/

    xla_gpu_lst = get_gpu_info()
    # declare the global queue
    # create the shared queue
    global task_queue
    task_queue = queue.Queue()
    global counter
    # shared counter for total tasks
    counter = threading.Semaphore(len(xla_gpu_lst))

    # issue all top-level tasks
    for i in range(20):
        task_queue.put((task_fun, (random.randint(1,5),)))

    # loop over all known tasks
    while True:
        # check for no further tasks
        with counter._cond:
            if not counter._value:
                time.sleep(0.5)
            else:
                # create a thread pool
                with mp.pool.ThreadPool(len(xla_gpu_lst)) as pool:
                    gpu = f'/gpu:{counter._value}'
                    # consume a task
                    try:
                        task, args = task_queue.get(timeout=0.5)
                    except queue.Empty:
                        break
                    # issue task to the thread pool
                    async_result = pool.apply_async(task(args), 
                            (gpu,), callback=callback)
                    # close the pool
                    pool.close()
                    # wait for all tasks to be processed
                    pool.join()

def worker(name):
    global task_queue
    while not task_queue.empty():
        gpu, duration, ind = task_queue.get(timeout=0.5)
        wait_time(name, duration, ind)
        callb(gpu)

def callb(arg):
    global task_queue, index_queue
    if not index_queue.empty():
        ind = index_queue.get()
        duration = random.randint(1,6)
        task_queue.put((arg, duration, ind))


def mat_mul(gpu):
    print(f'mat multi on {gpu}')
    # multiply two matrices on the `gpu` device
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

def long_mat_mul(gpu, duration):
    # multiply two matrices on the `gpu` device
    print(f'#######  in mat_mul {gpu} with duration {duration}')
    for _ in range(10_000*duration):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)

def worker_device(name):
    global task_queue
    time.sleep(5*random.random()) # sleep some random time
    with tf.device(name):
        while not task_queue.empty():
            gpu, duration, ind = task_queue.get(timeout=0.5)
            long_mat_mul(name, duration)# , duration, ind)
            callb(name)

def simple_worker_device(name):
    global index_queue
    with tf.device(name):
        while not task_queue.empty():
            gpu, duration, ind = index_queue.get(timeout=0.5)
            long_mat_mul(name, duration)# , duration, ind)

def main_w_dynamic_queue_worker():
    # declare and create a shared task queue
    gpus = get_gpu_info(q='GPU')
    #tf.debugging.set_log_device_placement(True)
    gpu_lst = list(range(2))
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
             tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print("###### ", len(gpus), 
                "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
        print(e)

    global task_queue, index_queue
    task_queue = queue.Queue()
    index_queue = queue.Queue()
    # Fill the index queue
    for i in range(20):
        index_queue.put(i)
    # Partially Fill the task queue
    for i in range(len(gpu_lst)):
        #duration = random.randint(1,15)
        duration = 5 if i%2==0 else 1
        gpu = "/gpu:{}".format(i%(len(gpu_lst)))
        ind = index_queue.get()
        task_queue.put((gpu, duration, ind))

    with mp.pool.ThreadPool(len(gpu_lst)) as pool:
        pool.map(worker_device, ['/gpu:'+repr(k) for k in range(len(gpu_lst))])

def main_w_dynamic_queue():
    # declare and create a shared task queue
    #gpu_lst = get_gpu_info()
    gpu_lst = list(range(3))

    global task_queue, index_queue
    task_queue = queue.Queue()
    index_queue = queue.Queue()
    # Fill the index queue
    for i in range(20):
        index_queue.put(i)
    # Partially Fill the task queue
    for i in range(len(gpu_lst)):
        #duration = random.randint(1,15)
        duration = 5 if i%2==0 else 1
        gpu = "/gpu:{}".format(i%(len(gpu_lst)))
        ind = index_queue.get()
        task_queue.put((gpu, duration, ind))

    while not task_queue.empty():
        with mp.pool.ThreadPool(len(gpu_lst)+1) as pool:
            #pool.starmap(task_fun, task_queue)
            while True:
                try:
                    gpu, duration, ind = task_queue.get(timeout=0.5)
                except queue.Empty:
                    break
                async_result = pool.apply_async(wait_time,
                        (gpu, duration, ind),
                        callback=callb)

            pool.close()
            pool.join()

def main_w_static_queue():
    # declare and create a shared task queue
    gpu_lst = get_gpu_info()
    global task_queue
    task_queue = queue.Queue()
    for i in range(20):
        #duration = random.randint(1,15)
        duration = 5 if i%2==0 else 1
        gpu = "/gpu:{}".format(i%(len(gpu_lst)+1))
        task_queue.put((gpu, duration))
    with mp.pool.ThreadPool(len(gpu_lst)+1) as pool:
        #pool.starmap(task_fun, task_queue)
        while True:
            try:
                gpu, duration = task_queue.get(timeout=0.5)
            except queue.Empty:
                break
            async_result = pool.apply_async(wait_time,
                    (gpu, duration),
                    callback=callb)

        pool.close()
        pool.join()

    
if __name__ == "__main__":
    main_w_dynamic_queue_worker()
    
