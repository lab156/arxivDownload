#!/usr/bin/env python3
from mpi4py import MPI
import multiprocessing as mp
import datetime as dt

def slow_fun(mach):
    """
    Takes about 4 sec on rpi
    """
    N = dt.datetime.now()
    print("At {}, machine {} started working"\
            .format((N.hour,N.minute,N.second), mach))
    sum([i*i for i in range(10_000_000)])
    return 0

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.Get_size()

    P = mp.Pool(processes=3)

    for i in range(10):
        if i%size == rank:
            #print("Machine {} got i = {}".format(rank, i))
            P.apply(slow_fun, (rank,))
    P.close()
    P.join()

if __name__ == "__main__":
    main()
