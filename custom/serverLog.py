import requests
from urllib.parse import quote
import _thread
import time
import threading

threadLock = threading.Lock()


def threadSendLog(content, name):
    threadLock.acquire()
    content = quote(content, 'utf-8')
    name = quote(name, 'utf-8')
    url = ''
    # print('sendLog:' + url)
    try:
        # print("----------------sendLog...----------------")
        r = requests.get(url, timeout=5)
        # print('\nsendLog finish', r.status_code, r.content)
        # print('sendLog finish')
    except Exception as e:
        print('\nsendLog network error!')
    finally:
        # print("----------------sendLog...----------------")
        threadLock.release()


class LogClass:
    def __init__(self, on=False):
        self.on = on

    def sendLog(self, content, name):
        if self.on:
            try:
                _thread.start_new_thread(threadSendLog, (content, name))
            except:
                print("cloud log error")


if __name__ == '__main__':
    log_class = LogClass()
    log_class.sendLog('35.8', 'PSNR')
    log_class.sendLog('35.8', 'PSNR')
    while True:
        pass
