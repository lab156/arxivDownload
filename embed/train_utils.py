from tensorflow.keras.callbacks import Callback
from datetime import datetime as dt


class TimeHistory(Callback):
    #def __init__(self):
    #    self.times = []
    #    super().__init__()
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = dt.now()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(dt.now() - self.epoch_time_start)
        #logs['ept'] = self.times

def scheduler(epoch, lr):
    if epoch < 5:
       return lr
    else:
       return lr * 0.5

def def_scheduler(decay):
    # RETURNS A VALID LEARNING RATE SCHEDULER
    def scheduler(epoch, lr):
        if epoch < 3:
            return lr
        else:
            return lr*decay
    return scheduler
