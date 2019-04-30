# coding=utf-8
import keras
import numpy as np
import os

from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ReduceLROnPlateau
from keras.models import Model
from keras import optimizers, regularizers
from keras import backend as K

stack_n = 5
layers = 6 * stack_n + 2
num_classes = 10
batch_size = 128
epochs = 200
iterations = 50000 // batch_size + 1
weight_decay = 1e-4

log_filepath = './my_resnet_32/'

def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test

def scheduler(epoch):
    # if epoch < 81:
    #     return 0.1
    # if epoch < 122:
    #     return 0.01
    # return 0.001
    # if epoch < 50:
    #     return 0.1
    # if epoch < 80:
    #     return 0.01
    # return 0.001
    if epoch < 20:
        return 0.1
    if epoch < 25:
        return 0.01
    if epoch < 30:
        return 0.005
    if epoch < 35:
        return 0.001
    if epoch < 40:
        return 0.0005
    if epoch < 45:
        return 0.0001
    if epoch < 50:
        return 0.00005
    return 0.000001

def residual_block(x, o_filters,increase=False):
    stride = (1, 1)
    if increase:
        stride = (2, 2)
    o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
    conv_1 = Conv2D(o_filters, kernel_size=(3, 3), strides=stride, padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(weight_decay))(o1)
    o2  = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
    conv_2 = Conv2D(o_filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizers.l2(weight_decay))(o2)
    if increase:
        projection = Conv2D(o_filters,kernel_size=(1, 1),strides=(2, 2),padding='same',
                            kernel_initializer="he_normal",
                            kernel_regularizer=regularizers.l2(weight_decay))(o1)
        block = add([conv_2, projection])
    else:
        block = add([conv_2, x])
    return block


def residual_network(img_input, classes_num=10, stack_n=5):
    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x, 16, False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x, 32, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 32, False)

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x, 64, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 64, False)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num, activation='softmax', kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x

def getResNetModel():
    img_input = Input(shape=(32, 32, 3))
    output = residual_network(img_input, 10, stack_n)
    resnet = Model(img_input, output)
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return resnet

def run():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)  # number of classes
    y_test = keras.utils.to_categorical(y_test, 10)  # number of classes

    # color preprocessing
    x_train, x_test = color_preprocessing(x_train, x_test)

    # img_input = Input(shape=(32, 32, 3))
    # # output = res_32(img_input)
    # output = residual_network(img_input, 10, stack_n)
    # resnet = Model(img_input, output)
    # sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    # resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    resnet = getResNetModel()
    resnet.summary()

    # set callback
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    lr_reduce = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    # cbks = [change_lr, tb_cb]
    cbks = [change_lr, lr_reduce]

    # dump checkpoint if you need.(add it to cbks)
    # ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto', period=10)

    # set data augmentation
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant', cval=0.)

    datagen.fit(x_train)

    # start training
    resnet.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                         steps_per_epoch=iterations,
                         epochs=epochs,
                         callbacks=cbks,
                         verbose=2,
                         validation_data=(x_test, y_test))
    resnet.save('my_resnet_32.h5')

if __name__ == "__main__":
    run()