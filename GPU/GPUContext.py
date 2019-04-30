import Application
import os
import sys
import threading
import re

class GPUDetectionThead(threading.Thread):

    def __init__(self, threadID, name, targetNode):
        threading.Thread.__init__(self)
        self.threadId = threadID
        self.name = name
        self.targetNode = targetNode

    def run(self):
        # print "Thread: " + str(self.threadId) + "  Target: " + str(self.targetNode)
        shellScript = 'ssh cu0' + str(self.targetNode) + ' "nvidia-smi"'
        output = os.popen(shellScript)
        file = output.readlines()

        data = []
        user = []
        for i in range(len(file)):
            if "N/A" in file[i]:
                data.append(file[i])
            # if "PID" in file[i]:

        for i in range(len(data)):
            print "\tcu0" + str(self.targetNode) + "\tGPU\t" + str(i) + "\t"+data[i]
        print "-----------------------------------------------------------------------------------------------------------"

def selectAllOfGPU():
    gpu_threads = []
    for i in range(5):
        tmp_thread = GPUDetectionThead((i + 1), "Thread" + str(i + 1), (i + 1))
        gpu_threads.append(tmp_thread)
    print "-----------------------------------------------------------------------------------------------------------"
    print "Target Node" + "\t" + " GPU" + "\t\t\t\t\t\t\t\t" + "Infomation"
    print "-----------------------------------------------------------------------------------------------------------"
    for i in range(len(gpu_threads)):
        gpu_threads[i].start()
        gpu_threads[i].join()

def selectNode(node_index=1):
    shellScript = 'ssh cu0' + str(node_index) + ' "nvidia-smi"'
    output = os.popen(shellScript)
    file = output.readlines()
    for i in range(len(file)):
        print file[i]

def run(cmd):
    output = os.popen(cmd)
    file = output.readlines()
    for i in range(len(file)):
        print file[i]

if __name__ == "__main__":

    selectAllOfGPU()
    #
    # selectNode(2)

    # run("sh ./startup.sh")


