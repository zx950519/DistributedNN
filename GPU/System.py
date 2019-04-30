# coding=utf-8
from itertools import chain
from collections import deque
from sys import getsizeof, stderr

import numpy as np
import gc
import sys
import datetime, time

# 垃圾回收
def garbageCollection(x):
    for i in range(x):
        gc.collect()

# print
def pout(output, rank):
    timestrp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print timestrp, "Print By", str(rank), output
    sys.stdout.flush()

def getTimeMilliseconds():
    return int(round(time.time() * 1000))

def totalSize(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:
        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}
    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            # print(s, type(o), repr(o), file=stderr)
            print(s, type(o), repr(o))

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)



# 聚合参数
def aggregationParameter(data, size):
    data = np.array(data)
    # print(data[0].shape)
    # print(data[0].shape[0])
    # print(data[0].shape[1])
    # print(data[0].shape[2])
    # print(data[0].shape[3])
    # exit(0)
    parameter = np.array([])
    for i in range(1, size):
        if(i==1):
            parameter = np.array([data[i-1][0]])
        else:
            parameter = np.concatenate((parameter, np.array([data[i-1][0]])), axis=0)
    parameter = parameter.sum(axis=0)
    parameter = parameter/(size-1)
    # print(parameter.shape)
    return parameter.tolist()

if __name__ == "__main__":
    timestrp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print timestrp