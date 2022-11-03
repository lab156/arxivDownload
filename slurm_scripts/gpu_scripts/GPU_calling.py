import tensorflow as tf
from datetime import datetime as dt

def burn_time(N = 500000):
    m1 = tf.constant([[0.0, 1.0],[-1.0, 0.0]])
    m2 = tf.constant([[1.0, 0.0],[0.0, 1.0]])
    for _ in range(N):
        m2 = m2*m1

def main():
    xla_gpu_lst = tf.config.list_physical_devices("XLA_GPU")
    print( "############################################################")
    print(f"##### The list of GPU devices found is: {xla_gpu_lst} ######")
    print( "############################################################")
    for k, gpu in enumerate(xla_gpu_lst):
        Now = dt.now()
        with tf.device(f"/gpu:{k}"):
            burn_time()
        burnt_time = (dt.now() - Now)
        print(f"Spent {burnt_time} burning time.")
    
if __name__ == "__main__":
    main()
    