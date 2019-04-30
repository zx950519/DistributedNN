# coding=utf-8
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical

import keras

def getCifar10Data(nb_Classes=10, fix_version=2):

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if fix_version == 1:
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
    elif fix_version == 2:
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
            x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]

    # 类别由单列转化为向量
    y_train = keras.utils.to_categorical(y_train, nb_Classes)
    y_test = keras.utils.to_categorical(y_test, nb_Classes)
    
    return x_train, y_train, x_test, y_test