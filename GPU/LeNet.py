# coding=utf-8
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras import backend as bk
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau


import keras
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 调整学习率
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print("Learning rate: ", lr)

    return lr

def getLeNetModel():
    # 序贯模型
    model = Sequential()
    # 卷积层-1
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                     input_shape=(28, 28, 1), padding="valid",
                     activation="relu", kernel_initializer="uniform"))
    # 池化层-1
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 卷积层-2
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1),
                     input_shape=(28, 28, 1), padding="valid",
                     activation="relu", kernel_initializer="uniform"))
    # 池化层-2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 展开层
    model.add(Flatten())
    # 全连接层
    model.add(Dense(100, activation="relu"))
    # 全连接层
    model.add(Dense(10, activation="softmax"))
    # compile
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=["accuracy"])
    # 模型概览
    # model.summary()
    # 返回模型
    return model

batch_size = 128    # 批处理大小
num_classes = 10    # 分类数
epoch = 50          # 迭代总数
img_rows = 28       # 图片高
img_cols = 28       # 图片宽
model_save_path = "./LeNet_model.json"    # 模型保存位置
weight_save_path = "./LeNet_weights.h5"   # 权重保存位置
data_augmentation = False                       # 不使用数据增强
checkpoint_path = "./model/lenet_minist_model.{epoch:03d}.h5"  # 回调函数记忆点的保存位置


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # 加载数据
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    print(y_train.shape)
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

    # 编译模型
    model = getLeNetModel()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=["accuracy"])
    # 模型概览
    # model.summary()

    # 设置回调函数,用于中间模型保存和学习率调整,仅保存最优模型
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True)
    # 学习率调度器
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=5,
                                   min_lr=0.5e-6)
    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # 训练模型
    if not data_augmentation:
        print("不使用数据增强")
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epoch,
                  verbose=1,
                  validation_data=(x_test, y_test))
    else:
        print("使用实时数据增强")
        datagen = ImageDataGenerator(
            # set input mean to 0 over the dataset
            featurewise_center=False,
            # set each sample mean to 0
            samplewise_center=False,
            # divide inputs by std of dataset
            featurewise_std_normalization=False,
            # divide each input by its std
            samplewise_std_normalization=False,
            # apply ZCA whitening
            zca_whitening=False,
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally
            width_shift_range=0.1,
            # randomly shift images vertically
            height_shift_range=0.1,
            # randomly flip images
            horizontal_flip=True,
            # randomly flip images
            vertical_flip=False)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                validation_data = (x_test, y_test),
                                epochs=epoch,
                                verbose=1,
                                workers=4,
                                callbacks=callbacks)

    # 统计得分
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # 保存模型及权重
    json_string = model.to_json()
    open(model_save_path, "w").write(json_string)
    model.save_weights(weight_save_path)