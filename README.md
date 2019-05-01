# Distributed Neural Network Framework
![](http://ww1.sinaimg.cn/large/005L0VzSly1g2lmyk95rbj30ku08caaa.jpg)  
[![experimental](http://badges.github.io/stability-badges/dist/experimental.svg)](http://github.com/badges/stability-badges)  
## :computer: 框架简介
分布式神经网络框架(Distributed Neural Network Framework)，简称DNNF，用于分布式环境下训练神经网络。随着机器学习训练数据集规模及模型参数数量不断增长，单机训练机器学习模型已不能适应大规模数据环境。近年来，随着硬件性能的飞速发展以及相关框架的火速推广，分布式机器学习框架获得了越来越多的关注。DNNF的开发重心在于针对异构场景下的深度学习，进行调度优化、聚合优化。DNNF支持：  
- 卷积神经网络的训练与预测
- CPU训练与GPU训练
- 异构场景下的自适应调度
- 自动调度训练

## :wrench: 环境&安装&使用
DNNF适用于分布式环境，可用于加速神经网络的训练，并保证模型性能损失在可容忍区间。下面给出环境设置、安装步骤以及使用方法。  
#### 环境
![](http://ww1.sinaimg.cn/large/005L0VzSly1g2lok40gu0j30z20jcmz9.jpg)

建议选择Linux作为分布式系统的操作系统。  
DNNF是一个用于训练神经网络的框架，因此需要Python环境。Python的版本随意，可以选择2.7.5或者3.+系列。这里推荐安装Anaconda，便于管理是安装新的依赖包。  
MPI(Message-Passing-Interface 消息传递接口)可实现是进程级别的通信，在进程之间进行消息传递。MPI在Linux系统下推荐安装OpenMP或MPICH版本。  
文件系统推荐使用NFS，这样集群仅需要一份代码，便于管理与维护。  

#### 安装
- Anaconda：https://blog.csdn.net/zhao12501/article/details/79832921  
- SSH免密：https://www.cnblogs.com/ivan0626/p/4144277.html
- NFS：https://zx950519.github.io/2017/06/05/CentOS7%E7%94%9F%E4%BA%A7%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA/
- MPI：https://blog.csdn.net/u014004096/article/details/50499429

#### 使用
```shell
cd your_project_path
sh ./startup.sh
```

## :coffee: 性能分析
建设中......  

## :watermelon: 致谢
- 感谢范雨辰同学的帮助

## :memo: 声明
本框架不是将网上的资料拼凑而来，而是参考了Keras与MPI的实例，结合实际训练中遇到的问题，为了解决问题而创作的框架。有任何疑问欢迎与作者交流。  
