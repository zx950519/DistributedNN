# coding=utf-8
import time, os, threading

def getTimeMilliseconds():
    return int(round(time.time() * 1000))

def timeStampToTime(timeStamp):
    timeStruct = time.localtime(timeStamp)
    return time.strftime('%Y-%m-%d %H:%M:%S', timeStruct)

def getFileSize(filePath):
    filePath = unicode(filePath, 'utf8')
    fsize = os.path.getsize(filePath)
    fsize = fsize/float(1024*1024)
    return round(fsize, 2)

def getFileAccessTime(filePath):
    filePath = unicode(filePath, 'utf-8')
    t = os.path.getatime(filePath)
    return timeStampToTime(t)

def getFileCreateTime(filePath):
    filePath = unicode(filePath, 'utf-8')
    t = os.path.getctime(filePath)
    return timeStampToTime(t)

def getFileModifyTime(filePath):
    filePath = unicode(filePath, 'utf-8')
    t = os.path.getmtime(filePath)
    return timeStampToTime(t)

def timeString2TimeStamp(str):
    timeArray = time.strptime(str, '%Y-%m-%d %H:%M:%S')
    return int(time.mktime(timeArray))

def timeStamp2TimeString(stamp):
    timeArray = time.localtime(stamp)
    return time.strftime('%Y-%m-%d %H:%M:%S', timeArray)

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
            if not os.path.exists("Task.log"):
                file = open("Task.log", "w")
                file.close()

            file = open("Task.log", "a")
            file.write(str(getFileModifyTime("Task.log"))+"\n")
            file.close()
            time.sleep(5)   # 心跳间隔

    def pause(self):
        self.__flag.clear()

    def resume(self):
        self.__flag.set()

    def stop(self):
        self.__flag.set()
        self.__running.clear()

        if os.path.exists("Task.log"):
            os.remove("Task.log")

if __name__ == "__main__":

    print getFileSize("./App.py")
    print getFileAccessTime("./App.py")
    print getFileCreateTime("./App.py")
    print getFileModifyTime("./App.py")

    # ipz = getTimeMilliseconds()
    # ipz = '2019-04-29 20:01:48'
    # print timeString2TimeStamp(ipz)
    # print timeStamp2TimeString(timeString2TimeStamp(ipz))

    jp = Job()
    jp.start()
    time.sleep(60)
    jp.stop()
