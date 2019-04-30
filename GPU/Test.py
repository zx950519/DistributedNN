import datetime, time, os, shutil
import threading
from Monitor import *

def swapProjects(file, dataSet, netWork):
    timeStrp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dirName = dataSet+"_"+netWork+"_"+timeStrp
    os.mkdir(dirName)
    for fi in file:
        shutil.move("./"+fi, dirName+"/"+fi)

class Job(threading.Thread):

    def __init__(self, *args, **kwargs):
        super(Job, self).__init__(*args, **kwargs)
        self.__flag = threading.Event()
        self.__flag.set()
        self.__running = threading.Event()
        self.__running.set()

    def run(self):
        while self.__running.isSet():
            self.__flag.wait()
            # print timeStamp2TimeString(getTimeMilliseconds())
            print time.time()
            time.sleep(1)

    def pause(self):
        self.__flag.clear()

    def resume(self):
        self.__flag.set()

    def stop(self):
        self.__flag.set()
        self.__running.clear()


if __name__ == "__main__":
    # file = ['mnist_LeNet_weights.h5']
    # swapProjects(file, 'mnist', 'LeNet_weights')

    a = Job()
    a.start()
    time.sleep(3)
    a.pause()
    time.sleep(3)
    a.resume()
    time.sleep(3)
    a.pause()
    time.sleep(3)
    a.stop()


