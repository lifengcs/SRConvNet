import time
import pytz
import time
import datetime


class RemainTime:
    def __init__(self, epoch):
        self.start_time = time.time()
        self.epoch = epoch

    def update(self, now_epoch):
        epoch_time = time.time() - self.start_time
        epoch_remaining = self.epoch - now_epoch
        time_remaining = epoch_time * epoch_remaining
        pytz.timezone('Asia/Shanghai')  # 东八区
        t = datetime.datetime.fromtimestamp(int(time.time()) + time_remaining,
                                            pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        print('epochs remaining:', epoch_remaining, '\tfinishing time:', t)

        self.start_time = time.time()
