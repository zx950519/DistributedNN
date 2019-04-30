# coding=utf-8
import math
import random
from System import *
from mpi4py import *
from Mnist import *
from Cifar10 import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import *
from LeNet import getLeNetModel
from DenseNet import getDenseNetModel
from ResNet import getResNetModel

# 超参
epochs = 1
ssp_value = 1
batch_size = 128
test_size = 0.05
vb = 2                                  # 训练可视化级别
minGap = 0                              # 最小随机数据增量
loadFactor = 0.05                       #
absoluteFactor = 0.8                    # 单节点绝对最小数据负载因子
enableRandomIncreation = False          # 是否启动随机数据生成
enableDataAugmentation = False          # 是否启用数据增强

def scheduler(epoch):
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

def readParameter():
    file = open("./config.ini", "r")
    dataSet = []
    netWork = []
    for line in file.readlines():
        if "data" in line:
            dataSet = line.strip().split(":")[1].split("#")
        if "network" in line:
            netWork = line.strip().split(":")[1].split("#")
    file.close()
    return dataSet[0], netWork[0]

def getSystemSetting():
    data = []
    data_disturibution = []
    data.append([])         # 0-训练集
    data.append([])         # 1-训练集标签
    data.append([])         # 2-测试集
    data.append([])         # 3-测试集标签
    data.append(epochs)     # 4-训练轮数
    datablocl_size = int(int(x_train.shape[0]) / (size - 1))
    data_disturibution_index = 0
    for i in range(0, size):
        if i == 0:
            data_disturibution.append(int(size - 1))
        else:
            tmp = []
            tmp.append(data_disturibution_index)
            tmp.append(datablocl_size)
            data_disturibution_index += datablocl_size
            data_disturibution.append(tmp)
    data.append(data_disturibution)  # 5-数据分块细节
    # pout("数据包总大小为: " + str(totalSize(data, verbose=False) / (1024 * 1024)) + "MB", rank)
    # pout("训练集数据包大小: " + str(totalSize(data[0], verbose=False) / (1024 * 1024)) + "MB", rank)
    # pout("训练集标签数据包大小: " + str(totalSize(data[1], verbose=False) / (1024 * 1024)) + "MB", rank)
    # pout("测试集数据包大小: " + str(totalSize(data[2], verbose=False) / (1024 * 1024)) + "MB", rank)
    # pout("测试集标签数据包大小: " + str(totalSize(data[3], verbose=False) / (1024 * 1024)) + "MB", rank)
    # pout("训练轮数数据包大小: " + str(totalSize(data[4], verbose=False) / (1024 * 1024)) + "MB", rank)
    # # pout("网络细节数据包大小: " + str(totalSize(data[5], verbose=False) / (1024 * 1024)) + "MB", rank)
    # pout("数据分块数据包大小: " + str(totalSize(data[5], verbose=False) / (1024 * 1024)) + "MB", rank)

    # data.append([])         # CUDA_GPU_Number

    pout("Total size: " + str(totalSize(data, verbose=False) / (1024 * 1024)) + "MB", rank)
    pout("Training data packet size: " + str(totalSize(data[0], verbose=False) / (1024 * 1024)) + "MB", rank)
    pout("Training data ticket size: " + str(totalSize(data[1], verbose=False) / (1024 * 1024)) + "MB", rank)
    pout("Testing data packet size: " + str(totalSize(data[2], verbose=False) / (1024 * 1024)) + "MB", rank)
    pout("Testing data ticket size: " + str(totalSize(data[3], verbose=False) / (1024 * 1024)) + "MB", rank)
    pout("Epochs: " + str(totalSize(data[4], verbose=False) / (1024 * 1024)) + "MB", rank)
    pout("Data Partition packet size: " + str(totalSize(data[5], verbose=False) / (1024 * 1024)) + "MB", rank)

    return data

def initHistorySetter():
    # 记录器
    history = []
    history.append([])  # 0-准确率
    history.append([])  # 1-Time
    history.append([])  # 2-Message
    history.append([])  # 3-NetWork IO
    history.append([])  # 4-Loss

    return history

def initServerMessage():
    gradient = []  # 最优梯度
    msg = []  # 消息数组
    msg.append(0)
    msg.append(gradient)
    return msg, gradient

def getCallBackList(version=1):
    checkpointer = ModelCheckpoint(monitor='val_acc', filepath="./checkpoint.txt", verbose=1, save_best_only=True,
                                   period=1)
    # lr_scheduler = LearningRateScheduler(lr_schedule)
    if version == 1:
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), # 旧版mnist+lenet高匹配
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)
        callbacks_list = [lr_reducer]   # 旧版mnist+lenet高匹配
    elif version == 2:
        change_lr = LearningRateScheduler(scheduler)
        lr_reduce = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        callbacks_list = [change_lr, lr_reduce]     # 新版cifar-10+resnet高匹配
    # callbacks_list = [checkpointer, lr_reducer]

    return callbacks_list

def getNetWork(net):
    if net == "LeNet":
        return getLeNetModel()
    # elif net == "VGG":
    #     return getVGGNet(shape_size=48)
    elif net == "DenseNet":
        return getDenseNetModel()
    elif net == "ResNet":
        return getResNetModel()

    return getLeNetModel()

def getTrainAndTestData(dataSet, netWork):
    if dataSet == "mnist":
        x_train, y_train, x_test, y_test = getMnistData()
    elif dataSet == "cifar10":
        x_train, y_train, x_test, y_test = getCifar10Data(nb_Classes=10, fix_version=2)  # 1:测试版 2:正式版
    return x_train, y_train, x_test, y_test

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    mission_time_start = getTimeMilliseconds()



    if (rank == 0):
        # 获取总进程数
        procs = size
        # 统计总流量
        total_Net_Cost = 0.0
        # 开始计时
        count_time_start = getTimeMilliseconds()
        # 读取配置文件
        dataSet, netWork = readParameter()
        pout("dataSet: "+dataSet+" netWork: "+netWork, rank)
        # 获取配置文件对应的数据集
        x_train, y_train, x_test, y_test = getTrainAndTestData(dataSet, netWork)
        # 分发数据准备
        data = getSystemSetting()
        count_time_end = getTimeMilliseconds()
        gap = count_time_end - count_time_start
        # pout("已完成数据分配, 耗时:" + str(float(gap / 1000)) + "秒", rank)
        pout("Data allocation has been completed, time-cost: " + str(float(gap / 1000)) + "seconds", rank)
        # 向计算进程发送消息
        send_time_start = getTimeMilliseconds()
        for i in range(1, size):
            comm.send(data, dest=i, tag=10001)
            total_Net_Cost = total_Net_Cost + totalSize(data)
        send_time_end = getTimeMilliseconds()
        send_tim_gap = send_time_end - send_time_start
        # pout("已完成数据发送, 耗时:" + str(float(send_tim_gap / 1000)) + "秒", rank)
        pout("Data have been sended, time-cost:" + str(float(send_tim_gap / 1000)) + "seconds", rank)
        # 记录器
        history = initHistorySetter()

        # 开始调度
        iter_tag = 20000    # 迭代所需标签
        msg, gradient = initServerMessage()     # 初始化消息&最优权值
        for j in range(0, epochs):
            iterator_time_start = getTimeMilliseconds()
            if j == 0:      # 分发权重
                msg[0] = 1
            else:
                msg[0] = 2
            # 接受聚合权值
            msg[1] = gradient
            # 发送消息
            for i in range(1, size):
                comm.send(msg, dest=i, tag=int(iter_tag + 0))
                total_Net_Cost = total_Net_Cost + totalSize(msg)
            receive_time_start = getTimeMilliseconds()
            # 回收消息
            backd = []
            for i in range(1, size):
                tmp = comm.recv(source=i, tag=int(iter_tag + 1))
                total_Net_Cost = total_Net_Cost + totalSize(tmp)
                backd.append(tmp)
            receive_time_end = getTimeMilliseconds()
            receive_time_gap = receive_time_end - receive_time_start
            # pout("已完成数据回收, 本轮训练总耗时:" + str(float(receive_time_gap / 1000)) + "秒", rank)
            pout("Data have been recycled, train time is: " + str(float(receive_time_gap / 1000)) + "seconds", rank)

            # 聚合权重
            # pout("开始聚合...", rank)
            pout("Start Aggregating...", rank)
            reconject_time_start = getTimeMilliseconds()
            # ToDo 不同的聚合策略
            gradient = aggregationParameter(backd, size)
            net_total_size = 0.0
            for i in range(1, size):
                net_total_size += totalSize(backd[i - 1], verbose=False)
                history[2].append(backd[i - 1])                                 #
            net_total_size = net_total_size / (1024 * 1024)
            # 测试聚合后的权重性能
            model = getNetWork(netWork)  # 获取指定网络
            model.set_weights(gradient)
            x_train, y_train, x_test, y_test = getTrainAndTestData(dataSet, netWork)
            score = model.evaluate(x_test, y_test, batch_size=128)
            # pout("聚合后的Loss: "+str(score[0]), rank)
            pout("Loss: "+str(score[0]), rank)
            # pout("聚合后的Accurary: "+str(score[1]), rank)
            pout("Accurary: "+str(score[1]), rank)

            reconject_time_end = getTimeMilliseconds()
            reconject_time_gap = reconject_time_end - reconject_time_start
            pout("Aggregation has been completed, time-cost: " + str(float(reconject_time_gap / 1000)) + "seconds", rank)
            pout("Epoch: " + str(j) + " end...  ", rank)
            iterator_time_end = getTimeMilliseconds()
            iterator_time_gap = iterator_time_end - iterator_time_start
            # 日志信息收集
            history[0].append(score[1])  # 0-本轮的准确率
            history[1].append(str(float(iterator_time_gap / 1000)) + "seconds")  # 1-本轮训练+聚合的时间开销
            history[3].append(str(net_total_size))  # 3-本轮训练产生的网络流量
            history[4].append(score[0])  # 4-本轮的Loss
            # ToDo 判断是否早停

            # 迭代标志自增
            iter_tag += 100

        # 发送停止指令
        for i in range(1, size):
            msg[0] = -9
            msg[1] = []
            comm.send(msg, dest=i, tag=int(iter_tag + 0))
        # ToDo 后处理
        # pout("开始后处理...", rank)
        # pout("后处理结束...", rank)
        # pout("本轮训练结束", rank)
        pout("*************************************************", rank)
        # 打印结果
        for i in range(0, epochs):
            pout("---------------------------------------------", rank)
            pout("Epoch " + str(i) + " accuracy:" + str(history[0][i]), rank)
            pout("Epoch " + str(i) + " time-cost:" + str(history[1][i]), rank)
            pout("Epoch " + str(i) + " network flow:" + str(history[3][i]+"MB"), rank)
            pout("Epoch " + str(i) + " Loss:" + str(history[4][i]), rank)
            pout("---------------------------------------------", rank)
        total_Net_Cost += totalSize(data)
        pout("Total network flow:" + str(total_Net_Cost / (1024 * 1024))+"MB", rank)

    else:
        dp = comm.recv(source=0, tag=10001)  # 接收数据
        comm.send(1, dest=0, tag=10003)  # 发送确认接受数据的信号
        # pout("训练集总规模 " + str(dp[0].shape), rank)
        # pout("训练集标签总规模 " + str(dp[1].shape), rank)
        # pout("测试集总规模 " + str(dp[2].shape), rank)
        # pout("测试集标签总规模 " + str(dp[3].shape), rank)
        pout("Epoch " + str(dp[4]), rank)
        pout("Data packet start index" + str(dp[5][rank][0]), rank)
        pout("Data packet end index" + str(int(dp[5][rank][0]) + int(dp[5][rank][1]) - 1), rank)
        pout("Data size" + str(dp[5][rank][1]), rank)

        # Set CUDA_GPU_Number
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(rank)%2)

        # 开始训练
        sub_tag = 20000
        while True:
            msg = comm.recv(source=0, tag=sub_tag)
            if msg[0] == -9:        # 判断是否收到kill信号
                break
            # 训练数据重构
            dataSet, netWork = readParameter()
            callbacks_list = getCallBackList(version=2)  # version=1:旧版mnist+LeNet适配 version=2:cifar10+ResNet适配
            x_train, y_train, x_test, y_test = getTrainAndTestData(dataSet, netWork)
            x_t, y_t, x_ts, y_ts = getTrainAndTestData(dataSet, netWork)
            sum = len(x_train)                  # 数据总量
            minGap = (int)(sum * loadFactor)    # 最小区间数据增量
            if (msg[0] != 1):
                # pout("非初始轮", rank)
                x_train = x_train[int(dp[5][rank][0]):int(dp[5][rank][0]) + int(dp[5][rank][1]) - 1, :]
                y_train = y_train[int(dp[5][rank][0]):int(dp[5][rank][0]) + int(dp[5][rank][1]) - 1]
                threshold = sum * absoluteFactor        # 获取单节点最小数据持有量
                if (len(x_train) < threshold and enableRandomIncreation == True):
                    delta = math.ceil(float(threshold - 1 - len(x_train)) / minGap)
                    # print(delta)
                    for k in range(0, delta):
                        start_index = random.randint(0, sum - 1 - minGap)
                        end_index = start_index + minGap
                        target_x = x_t[start_index:end_index, :]
                        target_y = y_t[start_index:end_index]
                        x_train = np.concatenate((x_train, target_x), axis=0)
                        y_train = np.concatenate((y_train, target_y), axis=0)
            else:
                pout("The First epoch", rank)
            # 构造网络与优化器
            model = getNetWork(netWork)                                             # 获取指定网络
            if msg[0] == 1:
                # pout("首次训练 无需加载权重", rank)
                pout("Do not load weights", rank)
            if msg[0] == 2:
                model.set_weights(msg[1])

            train_time_start = getTimeMilliseconds()

            if enableDataAugmentation:
                datagen = ImageDataGenerator(
                    featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                    zoom_range=0.1,  # Randomly zoom image
                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=False,  # randomly flip images
                    vertical_flip=False)  # randomly flip images
                datagen.fit(x_train)
                model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                    validation_data=(x_test, y_test),
                                    epochs=ssp_value,
                                    verbose=vb,
                                    steps_per_epoch=x_train.shape[0] // batch_size,
                                    callbacks=callbacks_list)
            else:
                model.fit(x_train, y_train,
                          validation_data=(x_test, y_test),
                          epochs=ssp_value,
                          verbose=vb,
                          batch_size=batch_size,
                          callbacks=callbacks_list)

            train_time_end = getTimeMilliseconds()
            train_time_gap = train_time_end - train_time_start
            # pout("已完成Train 耗时:" + str(float(train_time_gap / 1000)) + "秒", rank)
            pout("Training has been completed, time-cost:" + str(float(train_time_gap / 1000)) + "seconds", rank)
            test_time_end = getTimeMilliseconds()
            test_time_gap = test_time_end - train_time_end
            # pout("已完成Test 耗时:" + str(float(test_time_gap / 1000)) + "秒", rank)
            pout("Testing has been completed, time-cost:" + str(float(test_time_gap / 1000)) + "seconds", rank)
            # ToDo 抽取&保留权重
            # model.save_weights("./"+net_model_name+".hdf5")
            # 构造计算节点返回信息
            backd = []
            weights = model.get_weights()
            backd.append(weights)                               # 0 Weights
            backd.append(str(0))                                # 1 Loss
            backd.append(str(0))                                # 2 Accurary
            backd.append(str(float(train_time_gap / 1000)))     # 3 Train Time
            backd.append(str(float(test_time_gap / 1000)))      # 4 Test Time
            # 回传权重 数据包大小为
            comm.send(backd, dest=0, tag=int(sub_tag + 1))
            # pout("回传权重数据包大小为: " + str(totalSize(backd, verbose=False) / (1024 * 1024)) + "MB", rank)
            pout("The weights packet size : " + str(totalSize(backd, verbose=False) / (1024 * 1024)) + "MB", rank)
            sub_tag += 100

    mission_time_end = getTimeMilliseconds()
    mission_time_gap = mission_time_end - mission_time_start

    # pout("已完成本次任务，耗时:" + str(float(mission_time_gap / 1000)) + "秒", 0)
    pout("Mission completed，time-cost:" + str(float(mission_time_gap / 1000)) + "seconds", 0)