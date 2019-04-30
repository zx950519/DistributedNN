# coding=utf-8
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras import backend as bk
import keras

def getMnistData(img_rows=28, img_cols=28, num_classes=10):

    # 加载数据
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 判断是否需要修改通道位次
    if bk.image_data_format() == 'channels_first':
        print("channels_first")
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        print("channels_last")
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    print(x_train.shape)
    # 数据预处理
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    # 归一化
    x_train /= 255
    x_test /= 255

    # 转one-hot模式
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test

