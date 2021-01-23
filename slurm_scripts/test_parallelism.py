import multiprocessing as mp
from datetime import datetime, timedelta

def stupidfun(time_to_waste):
    cnt = 0
    Now = datetime.now()
    print("The function is starting, waiting for {}".format(time_to_waste))
    while (datetime.now() - Now).seconds < time_to_waste:
        cnt += 1
    return cnt

if __name__ == "__main__":
    #import sys
    #wait = int(sys.argv[1])
    #out = stupidfun(wait)
    with mp.Pool(processes=3, maxtasksperchild=1) as pool:
        pool.map(stupidfun, 3*[15])
    
