import multiprocessing as mp
import time
import itertools
from random import random

def sleep_wait(p):
    """
    just wait for t secs
    """
    gpu, t = p
    print(f'waiting for {t} seconds on {gpu}')
    time.sleep(t)
    return gpu

if __name__ == "__main__":
    gpu_lst = ['/gpu:0', '/gpu:1']
    gpu_iter = itertools.cycle(gpu_lst)

    gpu_states = mp.Array('i', [0 for _ in gpu_lst])

    pool_arg_lst = [(next(gpu_iter), t) for t in range(5,15)]
    #pool_task_lst = [(sleep_wait, (next(gpu_iter), t)) for t in range(5,15)]

    def pool_arg_it():
        global gpu_states
        for k in range(10):
            try:
                free_gpu = list(gpu_states).index(0)
                print(free_gpu, "\t", list(gpu_states))
                gpu_states[free_gpu] = 1
            except ValueError as e:
                print("no free gpus ", list(gpu_states))
            yield (f"/gpu:{free_gpu}", int(10*random()))

    with mp.Pool(2) as pool:
        #for ap in pool_arg_lst:
            #Res = pool.apply_async(sleep_wait, (next(gpu_iter), ap))
            #res2 = pool.apply(sleep_wait, (next(gpu_iter), ap))
            #res1.get()
            #res2.get()
        imap_it = pool.imap(sleep_wait, pool_arg_it())
        #res = pool.starmap_async(sleep_wait, pool_arg_it())
        for x in imap_it:
            freed_index = int(x[-1])
            gpu_states[freed_index] = 0
            print(x, "\t", list(gpu_states))
        #res.get()

    #proc_dict = {gpu: mp.Process for gpu in gpu_iter}
    #print(f"{proc_dict=}")
    #for a in pool_arg_lst:





