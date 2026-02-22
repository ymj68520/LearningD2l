# 跟着李沐学AI

## 目录

- [跟着李沐学AI](#跟着李沐学ai)
  - [目录](#目录)
  - [安装](#安装)
  - [数据操作](#数据操作)
    - [访问元素](#访问元素)
    - [tensor](#tensor)
  - [线性代数](#线性代数)
    - [torch线性代数](#torch线性代数)
    - [矩阵计算（求导）](#矩阵计算求导)
      - [自动求导(计算图)](#自动求导计算图)
  - [线性回归](#线性回归)
    - [数据处理](#数据处理)
    - [数据迭代指针](#数据迭代指针)
    - [初始化模型参数](#初始化模型参数)
    - [定义线性模型](#定义线性模型)
    - [平方损失函数](#平方损失函数)
    - [梯度下降](#梯度下降)
    - [训练](#训练)
    - [使用pytorch实现](#使用pytorch实现)
  - [Softmax回归](#softmax回归)
    - [损失函数](#损失函数)
    - [基本概念](#基本概念)
      - [核心思想：](#核心思想)
    - [数学公式](#数学公式)
      - [原始分数（Logits）](#原始分数logits)
      - [Softmax 变换](#softmax-变换)
    - [完整实现](#完整实现)
    - [使用pytorch](#使用pytorch)
  - [感知机（Perceptron）](#感知机perceptron)
    - [基本概念](#基本概念-1)
      - [核心思想：](#核心思想-1)
    - [数学模型](#数学模型)
      - [基本结构](#基本结构)
      - [数学公式](#数学公式-1)
    - [几何解释](#几何解释)
      - [分类规则：](#分类规则)
  - [多层感知机](#多层感知机)
    - [基本概念](#基本概念-2)
  - [网络结构](#网络结构)
    - [标准架构](#标准架构)
    - [数学表示](#数学表示)
    - [激活函数](#激活函数)
    - [代码实现](#代码实现)
  - [模型选择](#模型选择)
    - [误差](#误差)
    - [数据集](#数据集)
      - [K折交叉验证](#k折交叉验证)
    - [过拟合和欠拟合](#过拟合和欠拟合)
    - [模型复杂度](#模型复杂度)
    - [VC维](#vc维)
    - [权重衰退](#权重衰退)
    - [丢弃法](#丢弃法)
    - [数值稳定性](#数值稳定性)
      - [模型初始化](#模型初始化)
  - [神经网络](#神经网络)
    - [层和块](#层和块)
    - [参数管理](#参数管理)
    - [卷积](#卷积)
    - [卷积的相关代码实现](#卷积的相关代码实现)
      - [自定义层和块](#自定义层和块)
      - [参数管理](#参数管理-1)
      - [卷积的多输入和多输出](#卷积的多输入和多输出)
      - [卷积](#卷积-1)
      - [卷积的填充和步幅](#卷积的填充和步幅)
    - [池化](#池化)
      - [代码实现](#代码实现-1)
    - [LeNet](#lenet)
    - [AlexNet](#alexnet)
    - [VGG](#vgg)
    - [NiN](#nin)
    - [GoogLeNet](#googlenet)
    - [批量归一化](#批量归一化)
      - [核心思想](#核心思想-2)
      - [计算步骤](#计算步骤)
      - [代码实现](#代码实现-2)
    - [ResNet](#resnet)
      - [残差块](#残差块)
      - [ResNet块细节](#resnet块细节)
      - [代码实现](#代码实现-3)
    - [DenseNet](#densenet)
      - [核心理念](#核心理念)
      - [数学公式](#数学公式-2)
      - [网络架构](#网络架构)
        - [Dense Block (密集块)](#dense-block-密集块)
        - [Transition Layer (过渡层)](#transition-layer-过渡层)
        - [增长率 (Growth Rate, $k$)](#增长率-growth-rate-k)
        - [代码实现](#代码实现-4)
    - [DarkNet-53(yoloV3)](#darknet-53yolov3)
      - [核心设计理念](#核心设计理念)
      - [Darknet-53 的具体结构层级](#darknet-53-的具体结构层级)
      - [关键组件](#关键组件)
  - [多GPU训练](#多gpu训练)
    - [代码实现](#代码实现-5)
    - [使用pytorch库实现](#使用pytorch库实现)
    - [结果(单GPU)](#结果单gpu)
  - [分布式训练](#分布式训练)
    - [数据并行](#数据并行)
    - [同步SGD](#同步sgd)
  - [计算机视觉](#计算机视觉)
    - [数据增强](#数据增强)
      - [常见的图像数据增强方法](#常见的图像数据增强方法)
      - [代码实现](#代码实现-6)
      - [使用数据增强的训练代码](#使用数据增强的训练代码)
    - [微调（迁移学习）](#微调迁移学习)
      - [代码实现](#代码实现-7)
    - [物体检测](#物体检测)
      - [绘制框](#绘制框)
      - [数据集](#数据集-1)
      - [锚框](#锚框)
      - [R-CNN 区域卷积神经网络](#r-cnn-区域卷积神经网络)
        - [算法流程：](#算法流程)
        - [运作原理：](#运作原理)
        - [Faster R-CNN：](#faster-r-cnn)
        - [Mask R-CNN:](#mask-r-cnn)
        - [双线性插值](#双线性插值)
      - [SSD 单发多框检测](#ssd-单发多框检测)
        - [核心思想：单次检测 (Single Shot)](#核心思想单次检测-single-shot)
        - [两个技术创新](#两个技术创新)
        - [SSD 算法完整执行流程](#ssd-算法完整执行流程)
        - [代码实现(简化后仍复杂)](#代码实现简化后仍复杂)
      - [YOLO(You Only Look Once)](#yoloyou-only-look-once)
        - [工作原理](#工作原理)
        - [版本](#版本)
        - [V3结构](#v3结构)
    - [语义分割](#语义分割)
      - [数据集](#数据集-2)
      - [转置卷积](#转置卷积)
      - [全连接卷积神经网络(FCN)](#全连接卷积神经网络fcn)
    - [样式迁移](#样式迁移)
  - [RNN](#rnn)
    - [序列模型](#序列模型)
      - [N-Gram 模型 (统计学时代)](#n-gram-模型-统计学时代)
      - [隐马尔可夫模型 (HMM)](#隐马尔可夫模型-hmm)
    - [语言模型](#语言模型)
    - [N元语法(n-gram)](#n元语法n-gram)
      - [1元语法](#1元语法)
      - [2元语法](#2元语法)
      - [三元语法 (Trigram, N=3)](#三元语法-trigram-n3)
    - [循环神经网络](#循环神经网络)
      - [梯度剪裁](#梯度剪裁)
    - [GRU(门控循环单元)](#gru门控循环单元)
      - [详细流程和计算公式](#详细流程和计算公式)
    - [LSTM(长短期记忆网络)](#lstm长短期记忆网络)
      - [内部结构](#内部结构)
    - [深度循环神经网络](#深度循环神经网络)
    - [双向循环神经网络](#双向循环神经网络)
    - [编码器-解码器架构](#编码器-解码器架构)
    - [seq2seq](#seq2seq)
      - [BLEU(衡量生成序列好坏)](#bleu衡量生成序列好坏)
    - [束搜索](#束搜索)
      - [必要性](#必要性)
      - [算法流程](#算法流程-1)
      - [数学基础和数值稳定性](#数学基础和数值稳定性)
      - [长度偏好](#长度偏好)
  - [Transformer](#transformer)
    - [注意力机制](#注意力机制)
      - [Q、K、V](#qkv)
    - [NW核回归](#nw核回归)
    - [注意力评分函数](#注意力评分函数)
      - [概念](#概念)
      - [计算步骤](#计算步骤-1)
      - [掩蔽softmax操作](#掩蔽softmax操作)
      - [加性注意力](#加性注意力)
      - [缩放点积注意力](#缩放点积注意力)
    - [Bahdanau注意力](#bahdanau注意力)
      - [QKV](#qkv-1)
      - [举例说明](#举例说明)
    - [多头注意力](#多头注意力)
      - [必要性](#必要性-1)
      - [核心思想](#核心思想-3)
      - [算法流程](#算法流程-2)
    - [自注意力机制](#自注意力机制)
      - [必要性（解决指代问题）](#必要性解决指代问题)
      - [QKV](#qkv-2)
      - [位置编码](#位置编码)
      - [正弦余弦魔法](#正弦余弦魔法)
      - [位置拼接](#位置拼接)
    - [Transformer](#transformer-1)
      - [编码器](#编码器)
      - [解码器](#解码器)
  - [NLP](#nlp)
    - [预训练](#预训练)
      - [词嵌入(word2vec)](#词嵌入word2vec)
        - [one-hot编码的缺陷](#one-hot编码的缺陷)
        - [跳元模型（skip-gram）](#跳元模型skip-gram)
        - [连续词袋（CBOW）模型](#连续词袋cbow模型)
      - [近似训练](#近似训练)
      - [全局向量的词嵌入（GloVe）](#全局向量的词嵌入glove)
        - [数学原理](#数学原理)
        - [训练流程](#训练流程)
      - [子词嵌入](#子词嵌入)
        - [FastText](#fasttext)
        - [BPE](#bpe)
        - [BBPE](#bbpe)
        - [WordPiece](#wordpiece)

## 安装

首先需要安装conda

创建虚拟环境：

```bash
conda create -n name python-3.8 pip
conda activate name
```

安装必要的包（这里是CPU版本的，推荐安装GPU版本的）：

```bash
pip install jupyter d2l torch torchvision
```

课件所有代码：

```bash
curl https://zh-v2.d2l.ai/d2l-zh-2.0.0.zip -o d2l-zh.zip
unzip d2l-zh.zip && rm d2l-zh.zip
```

进入对应框架目录启动课件服务：

```bash
jupyter notebook
```

## 数据操作

N维数组是机器学习和深度学习的基本数据结构。

### 访问元素

* 一个元素：[1,2]
* 一行：[1,:]
* 一列：[:,1]
* 子区域：[1:3, 1:]
* 子区域：[::3. ::2]

### tensor

```python
x = torch.arange(12)
# 1 2 3 4 ... 12
# x = torch.arange(12)
x.shape
# torch.Size([12])
x.reshape(3,4)
# tensor([[ 0, 1, 2, 3], [ 4, 5, 6, 7], [ 8, 9, 10, 11]])
torch.zeros((2,3,4))

torch.ones((2,3,4))

torch.cat([x,y], dim=0) # concatenate along rows

torch.cat([x,y], dim=1) # concatenate along columns

before = id(y)

y = y + x

id(y) == before
# false

A = x.numpy()

B = torch.from_numpy(A)

type(A), type(B)

# 插值

import pandas as pd

data = torch.arange(12, dtype=torch.float32)

data = data.reshape(4, 3)

data = pd.DataFrame(data.numpy(), columns=['a', 'b', 'c'])

# print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]

# print(inputs)

# print(outputs)

inputs = inputs.fillna(inputs.mean()) # 用均值填充缺失值

# print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True) # 独热编码

# print(inputs)

x,y = torch.tensor(inputs.values), torch.tensor(outputs.values)

x,y
```

## 线性代数

略。。。。

### torch线性代数

```python
A = torch.arange(20).reshape(5,4)

A.T # Transpose

B = torch.tensor([[1,2,3],[2,0,4],[3,4,5]])

print(B)

B == B.T # Check symmetry

A = B.clone()

A.sum(axis=1) # Sum all elements

print(B.float().mean() ) # Mean of all elements 均值

C = B.sum()/B.numel() # Alternative way to compute mean

sum_B = B.sum(axis=1, keepdim=True)

print(sum_B)

B / sum_B # Normalize rows to sum to 1

B.cumsum(axis=0) # Cumulative sum along rows

# Linear algebra operations

x = torch.ones(3, dtype=torch.float32)

y = torch.randn((1,2,3), dtype=torch.float32)

x, y, # torch.dot(x, y) # Dot product

B.shape, x.shape, torch.mv(B.float(), x) # Matrix-vector product

torch.mm(A.float(), B.float()) # Matrix-matrix product

torch.norm(torch.ones(4, dtype=torch.float32)) # Vector norm

```

### 矩阵计算（求导）

亚导数。。

#### 自动求导(计算图)

一个函数在给定值上的导数值。

链式法则：

1. 正向累积
2. 反向累计，反向传递

```python
import torch

x = torch.arange(4.0)

x.requires_grad_(True) # Enable gradient tracking

xx = torch.arange(4.0, requires_grad=True) # Alternative way to enable gradient tracking

x.grad # Get the gradient

y = 2* torch.dot(x, x)

y.backward() # Compute gradients

x.grad # Print the gradient d y / d x

x.grad == 4 * x

x.grad.zero_() # Reset gradients

x.grad

y = x.sum()

y.backward()

x.grad.zero_() # Reset gradients

y = x * x

y.sum().backward()

x.grad.zero_() # Reset gradients

y = x*x

u = y.detach() # Detach y from the computational graph

z = u * x

z.sum().backward()

x.grad # Gradient only flows through x

# x.grad == u

x.grad.zero_() # Reset gradients

y.sum().backward()

x.grad # No gradient flow through u
```

## 线性回归

### 数据处理

```python
def synthetic_data(w, b, num_examples): #@save
    """Generate y = Xw + b + noise.
    w: weight vector
    b: bias term
    num_examples: number of examples to generate
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 查看数据
print('features:', features[0],'nlabel:', labels[0])

# features: tensor([0.1220, 0.1381]) label: tensor([3.9699])
```

### 数据迭代指针

```python
def data_iter(batch_size, features, labels):
    """
    batch_size: size of each mini-batch
    features: input features
    labels: corresponding labels
    """
    # The number of examples
    num_examples = len(features)
    # Create a list of indices and shuffle them
    indices = list(range(num_examples))
    random.shuffle(indices) # Shuffle the dataset
    # Generate mini-batches
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
        indices[i: min(i + batch_size, num_examples)])
        # Return the mini-batch of features and labels
        # like C++ iterators
        # every time we call data_iter, it yields a new mini-batch
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, 'n', y)   
    break
```

### 初始化模型参数

```python
# Initialize model parameters
# 2 features, so w has shape (2, 1)
# gradient tracking is enabled
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
# Initialize bias term
b = torch.zeros(1, requires_grad=True)
```

### 定义线性模型

```python
# Linear model

def linreg(X, w, b):
    """The linear regression model.    
    X: input features    
    w: weights    
    b: bias term
    """
    return torch.matmul(X, w) + b
```

### 平方损失函数

```python
# Squared loss function

def squared_loss(y_hat, y):
    """Squared loss function.    
    y_hat: predicted values    
    y: true values    
    """    
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

```

### 梯度下降

不断通过沿着梯度的反方向更新参数求解

小批量随即梯度，选取b个样本

两个超参数，学习率（损失函数中的步长），批量大小

```python
# optimizer: stochastic gradient descent
def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent.
    params: model parameters
    lr: learning rate
    batch_size: size of each mini-batch
    """
    with torch.no_grad():
        for param in params:
            # Update parameters using gradient
            # lr: learning rate
            # batch_size: to average the gradient
            param -= lr * param.grad / batch_size
            param.grad.zero_() # Reset gradients to zero after updating
```

### 训练

```python
lr = 0.03 # Learning rate
num_epochs = 3 # Number of epochs
net = linreg # Linear model
loss = squared_loss # Squared loss function

for epoch in range(num_epochs): # Loop over epochs
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # Compute loss
        l.sum().backward() # Backpropagate to compute gradients
        sgd([w, b], lr, batch_size) # Update parameters using SGD
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

输出：

```bash
epoch 1, loss 0.033290 
epoch 2, loss 0.000116 
epoch 3, loss 0.000049
```

误差打印：

```python
print(f"w的估计误差: {true_w - w.reshape(true_w.shape)}")
print(f"b的估计误差: {true_b - b}")

# w的估计误差: tensor([ 0.0002, -0.0008], grad_fn=<SubBackward0>) 
# b的估计误差: tensor([0.0002], grad_fn=<RsubBackward1>)
```

### 使用pytorch实现

```python
# Generate synthetic data
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# Create a data iterator

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.
    Args:
    data_arrays: List of data arrays.
    batch_size: Size of each mini-batch.
    is_train: Boolean indicating whether to shuffle the data.
    Returns:
    A DataLoader object for iterating over the dataset.
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))

# Define a linear regression model
# 2 features, 1 output
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
num_epochs = 3

for epoch in range(num_epochs):
	for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

## Softmax回归

交叉熵通常用来衡量两个概率的区别：

### 损失函数

```python
# 1. L2损失
L = 0.5 (y - y_hat)**2

# 2. L1损失
L = abs(y-y_hat)

# 3. Huber Robust
L = abs(y-y_hat) -0.5	# abs(y-y_hat) > 1
L = 0.5 (y - y_hat)**2  # otherwise
```

### 基本概念

**Softmax 回归**（也称为多项逻辑回归）是**逻辑回归的多类别推广**，用于解决**多分类问题**（大于2个类别）。

#### 核心思想：

* 将原始分数（logits）转换为**概率分布**

* 确保所有类别的概率之和为 1

* 每个类别的概率在 $[0, 1]$范围内

### 数学公式

#### 原始分数（Logits）

对于每个类别 $j$，计算得分：
$z_j​=w_j^T ​x+b_j​$

#### Softmax 变换

将得分转换为概率：
$P(y=j∣x)=\frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}$

其中：

* $K$：类别总数

* $z_j$：类别 $j$ 的得分

* 分母是所有类别得分的指数和

### 完整实现

```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 将图像展平的输入是长度为 784 的向量，输出是长度为 10（类别） 的向量
num_inputs = 784
num_outputs = 10

# w 的形状是 (784, 10)，b 的形状是 (10, )
# 使用正态分布初始化权重参数
# 标准差为 0.01，均值为 0
# requires_grad=True 表示需要计算梯度以进行反向传播
w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # 这里应用了广播机制

# 定义模型
def net(X):
    # X 展平成二维矩阵后，乘以权重矩阵，再加上偏差
    # -1: 表示这一维的大小由其他维度推断出来
    # 例如，假设 X 的形状是 (256, 1, 28, 28)，则展平后变成 (256, 784)
    # 然后与 w 矩阵相乘，得到形状为 (256, 10) 的矩阵，再加上偏差 b
    return softmax(torch.matmul(X.reshape((-1, w.shape[0])), w) + b)

# 实现交叉熵损失函数

def cross_entropy(y_hat, y):
	return -torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    """计算预测正确的数量（用于评价指标计算）
    参数:
    y_hat (torch.Tensor): 模型的输出。可以是：
        - 形状为 (n, c) 的二维张量，每行表示每个样本对各类别的得分或概率；
        - 或形状为 (n,) 的一维张量，表示已经是预测的类别索引。
    y (torch.Tensor): 真实标签，形状为 (n,)，每个元素是类别索引（整型）。
    返回:
    float: 预测正确的样本数量（Python 浮点数）。
    """
    # 如果 y_hat 是二维的且第二维大于1，说明每行是对各类别的分数/概率，
    # 需要取每行最大值对应的索引作为预测类别（axis=1 表示按列方向取最大值的索引）
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
    	y_hat = y_hat.argmax(axis=1)
    # 比较预测类别与真实标签，得到布尔张量 cmp（True 表示预测正确）
    cmp = y_hat.type(y.dtype) == y
    # 将布尔值转换为数值（0/1），并求和得到正确预测的数量，最后转换为 Python float 返回
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    """计算在某个数据集上的准确率
    参数:
    net: 要评估的模型，可以是自定义函数或 torch.nn.Module
    data_iter: 数据迭代器，返回每个批次的 (X, y)
    返回:
    准确率（正确预测样本数 / 总样本数）
    """
    # 如果 net 是 torch.nn.Module 的实例，则将其切换到评估模式（影响 dropout、batchnorm 等）
    # 不计算梯度以节省内存和计算
    if isinstance(net, torch.nn.Module):
    	net.eval() # 评估模式：关闭 dropout，使用固定的 batchnorm 统计
    # 评估模式 (net.eval())：关闭 dropout，batchnorm 使用累计统计；
    # 训练模式 (net.train())：开启 dropout，batchnorm 使用当前批次统计
    # 使用 d2l 提供的累加器记录两个量：
    # metric[0] 累加正确预测的样本数，metric[1] 累加预测的总样本数
    metric = d2l.Accumulator(2)
    # 遍历数据集中的每个批次
    for X, y in data_iter:
        # net(X) 得到模型对该批次的输出（可以是概率或分数矩阵）
        # accuracy(net(X), y) 计算该批次中预测正确的样本数（返回 float）
        # y.numel() 返回该批次中样本的数量（即标签个数）
        metric.add(accuracy(net(X), y), y.numel())
    # 返回在整个数据集上的准确率：正确预测总数 / 样本总数
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
    	net.train() # 训练模式：开启 dropout，batchnorm 使用当前批次统计
    metric = d2l.Accumulator(3) # 训练损失总和，训练准确率总和，样本数
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        
        if isinstance(updater, torch.optim.Optimizer):
            # 使用 PyTorch 内置的优化器和损失函数
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())
        else:
            # 使用自定义的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失平均值和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
        ylim=None, xscale='linear', yscale='linear',
        fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
        figsize=(6, 4)):
        # 增量地绘制多条线
        if legend is None:
        	legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
        	self.axes = [self.axes, ]
        # 设置坐标轴
        for ax in self.axes:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if legend:
            	ax.legend(legend)
        self.fmts = fmts
        self.X, self.Y = None, None
        
    def add(self, x, y):
        if not hasattr(y, "__len__"):
        	y = [y]
        n = len(y)
        if self.X is None:
        	self.X = [[] for _ in range(n)]
        if self.Y is None:
        	self.Y = [[] for _ in range(n)]
        for i in range(n):
            self.X[i].append(x)
            self.Y[i].append(y[i])
            self.axes[0].cla() # 清除当前轴
        for i in range(n):
        	self.axes[0].plot(self.X[i], self.Y[i], self.fmts[i])
        display.display(self.fig)
        display.clear_output(wait=True)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
    legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics

lr = 0.1
def updater(batch_size):
	return d2l.sgd([w, b], lr, batch_size)

num_epochs = 20
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

训练结果如下：
![softmax](./src/softmax1.svg)

预测：在测试集合上预测一次标签：
```python
def predict_ch3(net, test_iter, n=6):
    """预测标签"""
    for X, y in test_iter:
    	break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
    	X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n]
    )
    
predict_ch3(net, test_iter)
```
结果如下：
![预测](./src/softmaxpre1.svg)

### 使用pytorch

```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# pytorch 不会自动将二维张量转成一维张量，因此我们需要手动将其展平
# 这里我们使用 nn.Flatten 层来实现这一操作
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
def init_weights(m):
    if type(m) == nn.Linear:
    # 均值默认是0
    	nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```
## 感知机（Perceptron）

### 基本概念

**感知机**是**最简单的神经网络模型**，由 Frank Rosenblatt 在 1957 年提出。它是**二分类**的线性分类模型，是神经网络和支持向量机的基础。

#### 核心思想：

* 模仿生物神经元的工作原理

* 接受多个输入，产生一个输出

* 能够学习简单的线性决策边界

### 数学模型

#### 基本结构

一个感知机包含：

1. **输入层**：接收特征向量

2. **权重和偏置**：可学习参数

3. **激活函数**：阶跃函数（Step Function）

4. **输出层**：二分类结果

#### 数学公式

对于一个输入向量 $\mathbf{x} = [x_1, x_2, ..., x_n]$：

**加权和**：
$z=∑_{i=1}^n ​w_i ​x_i + b=w^Tx+b$

**激活函数（阶跃函数）**：

$f(z)=\begin{cases} 1 & if z >0 \\ 0 & otherwise \end{cases}​$

或使用符号函数：

$f(z)=sign(z)=\begin{cases} +1 & if z>0 \\ -1 & otherwise\end{cases}​$

### 几何解释

感知机实际上是在特征空间中寻找一个**超平面**：
$w^Tx+b=0$

* **权重向量 $\mathbf{w}$**：决定超平面的方向

* **偏置 $b$**：决定超平面的位置（偏移）

#### 分类规则：

* 如果 $\mathbf{w}^T \mathbf{x} + b > 0$，预测为正类（+1）

* 如果 $\mathbf{w}^T \mathbf{x} + b < 0$，预测为负类（-1）

## 多层感知机

### 基本概念

**多层感知机**（Multilayer Perceptron, MLP）是**单层感知机的扩展**，通过引入**隐藏层**和**非线性激活函数**，使其能够学习复杂的非线性模式。

## 网络结构

### 标准架构

```text
输入层 → 隐藏层1 → 隐藏层2 → ... → 隐藏层n → 输出层
     (n个神经元)    (m个神经元)            (k个神经元)
```

### 数学表示

对于一个 **L 层 MLP**：

**前向传播**：

1. 输入层：$\mathbf{h}^{(0)} = \mathbf{x}$

2. 隐藏层 $l$：$\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}$

3. 激活层：$\mathbf{h}^{(l)} = \sigma(\mathbf{z}^{(l)})$

4. 输出层：$\mathbf{y} = \text{softmax}(\mathbf{z}^{(L)})$（对于分类）

### 激活函数

激活函数引入**非线性**，是 MLP 的核心。

```python
# 1. Sigmoid（早期常用）
sigmoid = nn.Sigmoid()
# 公式：σ(x) = 1 / (1 + e^{-x})
# 问题：梯度消失、计算慢

# 2. Tanh
tanh = nn.Tanh()
# 公式：tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
# 输出范围：[-1, 1]，零中心化

# 3. ReLU（最常用）
relu = nn.ReLU()
# 公式：ReLU(x) = max(0, x)
# 优点：计算快、缓解梯度消失

# 4. Leaky ReLU
leaky_relu = nn.LeakyReLU(0.01)
# 公式：f(x) = x if x>0 else αx
# 解决"神经元死亡"问题

# 5. Softmax（输出层）
softmax = nn.Softmax(dim=1)
# 用于多分类，输出概率分布
```

### 代码实现

```python
net = nn.Sequential(
    nn.Flatten(), # flatten input
    nn.Linear(784, 256), # input layer
    nn.ReLU(), # hidden layer
    nn.Linear(256, 10) # output layer
)

def init_weights(m):
    if type(m) == nn.Linear:
    	nn.init.normal_(m.weight, std=0.01)
        
net.apply(init_weights)

batch_size = 256
lr = 0.1
num_epochs = 10

loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

结果如下：

![perceptron](./src/perceptron.png)

## 模型选择

### 误差

1. 训练误差：模型在训练数据上的误差
2. 泛化误差：模型在新数据上的误差

### 数据集

1. 训练数据集
2. 验证数据集

#### K折交叉验证

将数据集划分为 **K 个大小大致相等、互不重叠的子集（fold）**，每次选取其中 **1 个作为验证集，其余 K−1 个作为训练集**，重复 K 次，最终对性能指标取平均。

### 过拟合和欠拟合

| 维度            | 欠拟合         | 过拟合         |
| --------------- | -------------- | -------------- |
| 训练误差        | 高             | 低             |
| 验证 / 测试误差 | 高             | 高             |
| 泛化间隙        | 小（但整体差） | 大             |
| 学习曲线        | 早早停滞       | 后期分叉       |
| 偏差–方差       | 高偏差、低方差 | 低偏差、高方差 |

### 模型复杂度

参数个数，参数取值范围

### VC维

最大数据集容量：
单层感知机：VC维 = 3

### 权重衰退

使用均方范数，解决过拟合

一般来说$\eta * \lambda < 1$

### 丢弃法

在隐藏全连接层增加噪音，并且希望$E(x')= x$，对每个元素施加如下噪音：

$x'_i = \begin{cases} 0 & with\ probablity\ p \\ \frac{x_i}{1-p} & otherwise \end{cases}$

丢弃法相当于训练时的正则项，影响模型参数的更新；

在推理过程中，丢弃法直接返回输入

### 数值稳定性

梯度爆炸和梯度消失

* **梯度消失（Vanishing Gradient）**
  在反向传播时，梯度在多层网络中逐层相乘而不断减小，导致靠近输入层的参数几乎得不到有效更新，模型学习停滞或收敛极慢。

* **梯度爆炸（Exploding Gradient）**
  与之相反，梯度在传播过程中指数式放大，造成权重更新幅度过大，训练过程发散或出现数值溢出。

#### 模型初始化

将每一层的输出和梯度都看作随机变量 ,让每一层的方差是一个常数

1. 在合理的区间随机初始参数：训练开始容易有数值不稳定
2. Xavier初始：$n_{t-1} * \gamma_t = 1$，$n_t * \gamma_t=1$，但是如上两个条件很难全部满足，因此：
    $\gamma_t(n_{t-1}+n_t)/2 =1\rightarrow \gamma_t=2/(n_{t-1}+n_t)$
    正态分布：$\mathcal{N}(0,sqrt{2/(n_{t-1}+n_t)}$
    均匀分布：$\mathcal{U}(-\sqrt{6/(n_{t-1}+n_t)}, \sqrt{6/(n_{t-1}+n_t)})$，分布$\mathcal{U}[-a,a]$和方差为$\frac{a^2}{3}$
3. 适配权重形状变换，特别是$n_t$

## 神经网络

### 层和块

### 参数管理

### 卷积

处理图像的三通道信息，需要的神经元极多。

1. 平移不变性：二维卷积，全连接层的限制，抹去一些维度。
2. 局部性：评估输出时，不应选择远离输入的参数

对全连接层使用平移不变性和局部性得到卷积层。

卷积层将输入和核矩阵进行交叉相关，加上偏移后得到输出；核矩阵和偏移是可以学习的参数；核矩阵的大小是超参数。

填充和步幅是卷积层的超参数，填充是在周围添加一些额外的行列，控制输出的形状。

步幅是每次滑动窗口时的行列的步长，可以成倍减少输出的形状。

填充一般为：$p_h = k_h -1$，$p_w = k_w -1$；但是当核为奇数，上下两侧填充$\frac{p_h}{2}$，当核为偶数时，上侧$\lceil\frac{p_h}{2}\rceil$，下侧$\lfloor\frac{p_h}{2}\rfloor$

输入的高度和宽度都可以被步幅$s_h, s_w$整除，则输出大小为：
$(n_h\div s_h)\times(n_w\div s_w)$

### 卷积的相关代码实现

#### 自定义层和块

一般的一个nn模型使用Sequential构造：

```python
net = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
```

手动实现上述模型如下：

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
    	return self.out(F.relu(self.hidden(X)))

net = MLP()

class MySequential(nn.Module):
    def __init__(self, *args):
    	super().__init__()
    	for block in args:
    		self._modules[block] = block
    
    def forward(self, X):
    	for block in self._modules.values():
    		X = block(X)
    	return X

net = MySequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
```

在自己实现的MLP中手动加入一个隐藏层

```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)
    
    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.matmul(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
        	X /= 2
        return X.sum()
net = FixedHiddenMLP()
```

合并后简洁实现：

```python
class NestedMLP(nn.Module):
def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU()
    )
    self.out = nn.Linear(32, 16)

def forward(self, X):
	return self.out(self.net(X))
    
chimera = nn.Sequential(
    NestedMLP(),
    nn.Linear(16, 20),
    FixedHiddenMLP()
)


print(chimera(X))
# tensor(0.3952, grad_fn=<SumBackward0>)
```
#### 参数管理

考虑在有module的情况下访问或者管理参数。

首先考虑一个单隐藏层MLP：

```python
net = nn.Sequential(
    nn.Linear(4,8),
    nn.ReLU(),
    nn.Linear(8,1)
)
```

获取其中一层的所有参数：

```python
print(net[2].state_dict()) # 访问第二层的参数
# OrderedDict([('weight', tensor([[ 0.1968, 0.2480, 0.3306, 0.1889, -0.2792, 0.0940, 0.2401, 0.1442]])), ('bias', tensor([0.0610]))])
```

获取某一层的偏置参数：

```python
# 访问第二层的偏置参数
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

# <class 'torch.nn.parameter.Parameter'> 
# Parameter containing: 
# tensor([0.0610], requires_grad=True) 
# tensor([0.0610])
```

权重的梯度在反向传播前为空：

```python
print(net[2].weight.grad == None)

# True
```

一次性访问所有参数：

```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])

print(*[(name, param.shape) for name, param in net.named_parameters()])


# ('weight', torch.Size([8, 4])) ('bias', torch.Size([8])) 
# ('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
```

直接访问命名参数：

```python
net.state_dict()['2.bias'].data

# tensor([0.0610])
```

可以直接打印网络结构，首先定义一个网络如下：

```python
def block1():
    return nn.Sequential(
        nn.Linear(4,8),
        nn.ReLU(),
        nn.Linear(8,4),
        nn.ReLU()
    )

def block2():
    net = nn.Sequential()
    for i in range(4):
    	net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(
    block2(),
    nn.Linear(4,1)
)
```

打印网络的结构：

```python
print(rgnet)
```

网络的结构显示如下：

```python
Sequential(
  (0): Sequential(
    (block 0): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 1): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 2): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 3): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
  )
  (1): Linear(in_features=4, out_features=1, bias=True)
)
```

修改默认的初始参数：

```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
        
net.apply(init_normal)
print(net[0].weight.data[0])  # 查看第一层的权重参数
print(net[0].bias.data)      # 查看第一层的偏置参数

# tensor([-0.0051, -0.0083,  0.0073, -0.0136])
# tensor([0., 0., 0., 0., 0., 0., 0., 0.])
```

使用xavier初始化参数：

```python
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])  # 查看第一层的权重参数
print(net[2].weight.data)      # 查看第二层的权重参数

# tensor([ 0.1444, -0.5032,  0.3079, -0.4662])
# tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])
```

自定义初始化函数：

```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5
        
net.apply(my_init)
print(net[0].weight.data)  # 查看第一层的权重参数
```

输出结果如下：

```python
Init ('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
Init ('weight', torch.Size([1, 8])) ('bias', torch.Size([1]))
tensor([[ 0.0000, -0.0000,  0.0000, -0.0000],
        [-0.0000,  0.0000, -0.0000, -0.0000],
        [ 0.0000,  0.0000,  0.0000, -0.0000],
        [ 9.0355,  0.0000, -0.0000, -8.7955],
        [ 9.7313, -8.7391,  5.5579,  0.0000],
        [-5.8401,  0.0000, -9.5707, -0.0000],
        [-9.2654,  6.4986, -8.7337,  5.2166],
        [-5.4260, -8.8009,  0.0000,  0.0000]])
```

可以使用一种简单粗暴的手段初始化：

```python
net[0].weight.data[:] += 1.0
net[0].weight.data[0, 0] = 42
print(net[0].weight.data[0])

# tensor([42.,  2.,  2.,  2.])
```

参数绑定，也就是多个层共享一些参数：

```python
shared = nn.Linear(8,8)
net = nn.Sequential(
    nn.Linear(4,8),
    nn.ReLU(),
    shared,
    nn.ReLU(),
    shared,
    nn.ReLU(),
    nn.Linear(8,1)
)

net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])  # True
net[2].weight.data[0,0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])  # True
```

也可以自定义一个无参数的层：

```python
# 自定义一个无参数的层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return X - X.mean()
    
layer = CenteredLayer()
print(layer(torch.FloatTensor([1,2,3,4,5])))  # tensor([-2., -1., 0., 1., 2.])

net = nn.Sequential(
    nn.Linear(8,128),
    CenteredLayer()
)
Y = net(torch.rand(4,8))
print(Y.mean())  # tensor(-7.4506e-09, grad_fn=<MeanBackward0>)
```

定义一个有参数的层：

```python
# 自定义一个有参数的层
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features,))
        # self.weight = nn.Parameter(torch.zeros(in_features, out_features))
        # self.bias = nn.Parameter(torch.zeros(out_features,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
    
dense = MyLinear(5,3)
print(dense.weight)
```

使用自定义的层执行正向传播计算：

```python
print(dense(torch.rand(2,5)))

# tensor([[0.7151, 1.4433, 0.0000],
#         [1.3851, 2.3780, 0.0000]])
```

#### 卷积的多输入和多输出

二维卷积的多输入多输出的不同组合：

```python
def corr2d_multi_in(X, K):
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
```

1x1卷积，即全连接层，进行验证：

```python
# 1X1 卷积
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))


X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
```

#### 卷积

手动定义一个二维卷积函数：

```python
def corr2d(X, K):
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y
```

定义一个二维卷积块：

```python
class conv2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

简单使用这个二维卷积做训练：

```python
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y)**2
    conv2d.zero_grad()
    l.sum().backward()
    # 手写实现SGD
    # 访问权重参数的梯度，使用学习率3e-2更新权重参数
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad.data
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
```

输出如下：

```python
epoch 2, loss 1.719
epoch 4, loss 0.304
epoch 6, loss 0.057
epoch 8, loss 0.012
epoch 10, loss 0.003
```

#### 卷积的填充和步幅

使用pytorch的Conv2d参数调整：

```python
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
```

### 池化

池化层类似卷积，但是执行的操作不再是交叉相关操作；池化曾返回窗口中最大或者平均值。

池化可以缓解卷积层对于位置信息的敏感性，且且同样有窗口大小、填充和步幅作为超参数。

#### 代码实现

手动实现一个正向传播二维池化：

```python
# 正向传播实现二维池化层
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = torch.max(X[i:i + p_h, j:j + p_w])
            elif mode == 'avg':
                Y[i, j] = torch.mean(X[i:i + p_h, j:j + p_w])
    return Y
```

使用pytorch的函数：

1. max池化：`pool2d = nn.MaxPool2d(3, stride=2, padding=1)`
2. avg池化： `pool2d = nn.AvgPool2d(3, stride=2, padding=1)`

### LeNet

实现LeNet,并在Fashion Mnist上训练测试：
（使用Lazy*函数避免手动计算输出输出维度）

```python
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

net = nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),  # nn.Sigmoid()改为ReLU以改善梯度流动，避免Sigmoid的梯度消失问题
    # nn.AvgPool2d(kernel_size=2, stride=2),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),  # 同样改为ReLU
    # nn.AvgPool2d(kernel_size=2, stride=2),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    # nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.LazyLinear(120), nn.ReLU(),  # 改为ReLU
    nn.Linear(120, 84), nn.ReLU(),  # 改为ReLU
    nn.Linear(84, 10)
)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

def evaluate_accuracy(net, data_iter, device = None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        if isinstance(X, list):
            # BERT微调所需（之后将介绍）
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
        
    return metric[0] / metric[1]

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型（在第6章中将介绍）"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer = d2l.Timer()
    num_batches = len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失总和、训练准确率总和、样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if( i+1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.6f}, train acc {train_acc:.6f}, '
          f'test acc {test_acc:.6f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')

lr, num_epochs = 0.05, 20 # 低学习率适配ReLU, 增加epoch以获得更好收敛

train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

```

训练结果如下：

```python
loss 0.276554, train acc 0.898100, test acc 0.870700 
143448.7 examples/sec on cuda:0
```

![LeNet](./src/LeNet.svg)

### AlexNet

相比LeNet, 将卷积层增加到5层，全连接层增加到3层。

具体代码如下：

首先定义这个网络：

```python
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), 
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), 
    nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), 
    nn.ReLU(),
    # nn.Conv2d(384, 256, kernel_size=3, padding=1), 
    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)
```

定义batch size和学习参数，然后训练：

```python
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs=num_epochs, lr=lr, device=d2l.try_gpu())
```

不做调参，简单训练结果如下：

```python
loss 0.330, train acc 0.881, test acc 0.882 
3358.1 examples/sec on cuda:0
```

![AlexNet](./src/AlexNet.svg)

### VGG

AlexNet使用较大的卷积层，消耗很高。分VGG块，3x3卷积，n层，m通道，2x2最大池化层。

代码如下：

首先定义VGG块：

```python
import torch
from torch import nn
from d2l import torch as d2l

def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for _ in range(num_convs):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)
```

定义经典的网络结构：

```python
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
```

定义网络：

```python
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    
    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )

net = vgg(conv_arch)
```

为了方便训练，缩小网络结构：

```python
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
```

定义训练参数，开始训练：

```python
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs=num_epochs, lr=lr, device=d2l.try_gpu())
```

训练较慢，3080花费5m43s，训练结果如下：

```python
loss 0.170, train acc 0.937, test acc 0.919
1985.9 examples/sec on cuda:0
```

![VGG](./src/VGG.svg)

### NiN

卷积层需要的参数比较少，但是卷积层后的第一个全连接层所需的参数极其庞大，LeNet：48k, AlexNet：26M, VGG：102M.

NiN块：一个卷积层后跟两个全连接层。步幅1, 无填充，全连接层为1x1卷积。

NiN无全连接层，交替使用NiN块和步幅为2的最大池化，逐步减小高宽和增大通道数。最后使用全局平均池化得到输出，其输入通道数是类别数。

代码实现如下：

```python
import torch
from torch import nn
from d2l import torch as d2l

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

lr, num_epochs = 0.05, 10
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs=num_epochs, lr=lr, device=d2l.try_gpu())
```

训练结果如下：

```python
loss 0.355, train acc 0.870, test acc 0.856
2681.8 examples/sec on cuda:0
```

![NiN](./src/NiN.svg)

### GoogLeNet

Inception块：四个路径从不同层面抽取信息，然后在输出通道维合并。

1. 1 Conv
2. 1 Conv + 3 Conv,pad 1
3. 1 Conv + 5 Conv,pad 2
4. 3 MaxPool, pad1 + 1 Conv

跟单3x3或者5x5卷积相比，Inception具有更少的参数个数和计算复杂度。

InceptionV3：

1. 将stage3的5 Conv改为两个3 Conv
2. 将stage4的3 Conv改为一个1x7 Conv和一个7x1 Conv；再将5 Conv改为两组，每组一个1x7 Conv和一个7x1 Conv
3. 将stage5的3 Conv改为并行的两个，3x1 Conv和1x3 Conv；将5 Conv改为一个3 Conv后接两个并行的3x1 Conv和1x3 Conv。

V1原始版本的实现如下，首先定义Inception块：

```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 1x1 conv branch
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 1x1 conv -> 3x3 conv branch
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 1x1 conv -> 5x5 conv branch
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 3x3 max pooling -> 1x1 conv branch
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)
```

上述块照抄即可，定义每一个stage如下：

```python
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
b3 = nn.Sequential(
    Inception(192, 64, (96, 128), (16, 32), 32),
    Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
b4 = nn.Sequential(
    Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
b5 = nn.Sequential(
    Inception(832, 256, (160, 320), (32, 128), 128),
    Inception(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
```

开始训练：

```python
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

训练结果如下：

```python
loss 0.241, train acc 0.908, test acc 0.895
2919.5 examples/sec on cuda:0
```

![GoogLeNet](./src/GoogleNet.svg)

### 批量归一化

损失出现在最后，后面的层训练较快。

数据在最底部，底部的层训练较慢，底部层一变化，所有都得跟着变，最后的那些层需要重新学习很多次，导致收敛变慢。

#### 核心思想

对神经网络中每一层的输入（或输出）进行归一化，使其分布保持稳定（均值为0，方差为1），从而减少内部协变量偏移（Internal Covariate Shift，即每层输入分布因前层参数更新而不断变化的问题）。

#### 计算步骤

假设一个批次（batch）的输入为 $x\in \mathbb{R}^{B\times C}$（B 为批次大小，C 为特征维度），BN的步骤如下：

1. 计算批次均值和方差：
$\mu_B = \frac{1}{B}\sum_{i=1}^{B} x_i,\qquad \sigma_B^2 = \frac{1}{B}\sum_{i=1}^{B}(x_i - \mu_B)^2$
2. 归一化：
$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$其中，$\epsilon$是小常数，防止除零
3. 缩放和平移（引入可学习参数 γ 和 β ）：
$y_i = \gamma\hat{x}_i + \beta$

其中可学习的参数为$\lambda$和$\beta$，作用在全连接层和卷积层输出上，激活函数前，或者作用在全连接层和卷积层输入上；对于全连接层，作用在特征维；对于卷积层，作用在通道维。

#### 代码实现

```python
import torch
from torch import nn
from d2l import torch as d2l

# batch normalization function
# X: input tensor
# gamma: scale parameter
# beta: shift parameter
# moving_mean: running mean for inference
# moving_var: running variance for inference
# eps: small constant to avoid division by zero
# momentum: momentum for running mean/variance, used during training to update moving averages
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        # 推理模式：训练期间累积的滑动平均均值/方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 简单起见，我们假设输入为2D或4D张量（全连接层或卷积层的输入）
        assert len(X.shape) in (2, 4)
        # 全连接层，计算特征维度上的均值和方差
        if len(X.shape) == 2:
            # mean和var的形状与X的特征维度相同
            # dim=0表示按行计算均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 卷积层，计算通道维度上的均值和方差
            # dim=(0,2,3)表示按批量和空间维度计算均值和方差
            # keepdim=True保持均值和方差的维度，以便后续广播运算
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式：使用当前批次的均值和方差进行归一化
        # X_hat: 用当前batch的均值和方差归一化到分布N(0,1)【标准正态分布】
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 把当前batch算出的均值和方差，按照动量累积到moving_mean和moving_var中
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    # BN的仿射变换
    # 由于之前的归一化把数据变成了标准正态分布N(0,1)，所以需要通过gamma和beta进行仿射变换，恢复出数据的表达能力
    Y = gamma * X_hat + beta
    return Y, moving_mean, moving_var

# Batch Normalization Layer
class BatchNorm(nn.Module):
    # num_features: number of features (output dimension of fully connected layer or number of channels in convolutional layer)
    # num_dims: dimension of input data (2 for fully connected layer, 4 for convolutional layer)
    def __init__(self, num_features, num_dims):
        super().__init__()
        # num_features: 特征数量（全连接层的输出维度或卷积层的通道数）
        # num_dims: 输入数据的维度（2表示全连接层，4表示卷积层）
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 可学习的缩放参数gamma和偏移参数beta
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 用于推理的移动平均均值和方差，不作为模型参数参与梯度更新
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 调用batch_norm函数进行批量归一化
        Y, self.moving_mean, self.moving_var = batch_norm(
            X,
            self.gamma,
            self.beta,
            self.moving_mean,
            self.moving_var,
            # 避免除零错误的小常数, 不同框架可能使用不同的默认值
            eps=1e-5,
            momentum=0.9,
        )
        return Y

# Example: Using BatchNorm in LeNet
# 通常在卷积层后和激活函数前使用批量归一化
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5),
    BatchNorm(6, num_dims=4),
    # nn.Sigmoid(),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    BatchNorm(16, num_dims=4),
    # nn.Sigmoid(),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 120),
    BatchNorm(120, num_dims=2),
    # nn.Sigmoid(),
    nn.ReLU(),
    nn.Linear(120, 84),
    # 最后输出层通常不使用批量归一化，但是可以根据需要添加
    # BatchNorm(84, num_dims=2),
    # nn.Sigmoid(),
    nn.ReLU(),
    nn.Linear(84, 10),
)
```

训练结果如下：

```python
lr, num_epochs, batch_size = 0.5, 20, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

```python
loss 0.126, train acc 0.952, test acc 0.893
83944.7 examples/sec on cuda:0
```

![BN](./src/BN.svg)

### ResNet

添加更多的神经网络的层数，不一定总是改进精度，有可能出现模型偏差。

#### 残差块

残差块（Residual Block）是 ResNet 的核心构件，它把“普通堆叠”变成“带捷径的残差学习”。

输入张量 x 从左边进来，分成两路：

1. 主径（residual path，要学的部分）
   x → W₁ → BN → ReLU → W₂ → BN → ReLU → W₃ → BN → F(x)
   注：W₁、W₂、W₃ 代表 1×1→3×3→1×1 三层卷积，合起来叫“残差函数”F(x)。
2. 捷径（shortcut path，恒等或投影）
   x ────────────→ identity / projection ───────────→
   若形状完全一致，什么参数都不加；
   若通道数或尺寸对不上，就插一个 1×1 卷积（stride=2）把 x 投影成与 F(x) 同形，记作 W_s·x。
3. 合并门（addition）
   把两路逐元素相加：y = F(x) + x（或 y = F(x) + W_s·x）。
4. 激活门（post-ReLU）
   相加结果再过一个 ReLU，得到残差块的最终输出 y，同时也作为下一块的输入。

残差块=“主径学残差，捷径保恒等，相加后激活”

#### ResNet块细节

具体设计参考VGG，具体如下：

1. 块内不变换通道：X -> 3 Conv -> BN ->ReLU -> 3 Conv -> BN -> +X -> ReLU
2. 块内变换通道(1 Conv)：X -> 3 Conv -> BN ->ReLU -> 3 Conv -> BN -> + 1 Conv(X) -> ReLU

#### 代码实现

引入相关库：

```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

定义残差网络：

```python
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3,
                               padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                               padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)
```

定义整个ResNet的5个stage：

```python
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residual(input_channels, num_channels, use_1x1conv=True, strides=2)
            )
        else:
            blk.append(Residual(num_channels, num_channels))
    return nn.Sequential(*blk)


b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(
    b1,
    b2,
    b3,
    b4,
    b5,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 10),
)
```

开始训练：

```python
lr, num_epochs, batch_size = 0.01, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

训练结果如下：

```python
loss 0.148, train acc 0.950, test acc 0.855
872.2 examples/sec on cuda:0
```

![ResNet](./src/ResNet.svg)

### DenseNet

DenseNet 的核心思想是对卷积神经网络中信息的流动方式进行了彻底的重新思考。

#### 核心理念

极致的连接 (Dense Connectivity)

在 DenseNet 出现之前，主流的卷积神经网络（如 VGG, ResNet）主要通过层与层之间的串联来传递信息。

* 传统 CNN: 第 $L$ 层只接收第 $L-1$ 层的输出。
* ResNet (残差网络): 通过恒等映射（Identity Shortcut），将输入与输出相加 (Add)。公式为：$x_l = H_l(x_{l-1}) + x_{l-1}$。

DenseNet 的创新点：DenseNet 提出了一种密集连接机制。在一个被称为 "Dense Block" 的模块内，每一层都与所有之前的层直接相连。这意味着，第 $l$ 层接收前面所有层 ($0, 1, ..., l-1$) 的特征图（Feature Maps）作为输入。

#### 数学公式

$$x_l = H_l([x_0, x_1, ..., x_{l-1}])$$

其中：

* $[x_0, x_1, ..., x_{l-1}]$ 表示将之前所有层的特征图在通道维度 (Channel axis) 上进行拼接 (Concatenation)。
* $H_l$ 是一个非线性变换函数（通常包含 BN + ReLU + Conv）。

#### 网络架构

DenseNet 主要由两个核心模块组成：Dense Block 和 Transition Layer。

整体架构如下：

![densenet-1](./src/denseNet-1.svg)

Dense Block 内部细节:

![denseNet-2](./src/denseNet-2.svg)

##### Dense Block (密集块)

这是网络的核心组件。在一个 Block 内部，特征图的尺寸（长和宽）保持不变，以便于进行通道拼接（Concatenation）。

* 特征重用 (Feature Reuse): 由于每一层都能“看到”之前所有的特征，网络可以更高效地利用低级特征，而不需要重复学习。
* 瓶颈层 (Bottleneck Layers): 为了减少计算量，DenseNet 通常在 $3\times3$ 卷积之前引入一个 $1\times1$ 卷积（称为 DenseNet-B）。结构通常是：`BN -> ReLU -> Conv(1x1) -> BN -> ReLU -> Conv(3x3)`。

##### Transition Layer (过渡层)

由于 CNN 需要不断缩小特征图的尺寸（Downsampling）来提取高层语义，而 Dense Block 内部保持尺寸不变，因此需要 Transition Layer 连接两个 Dense Block。

* 作用： 降低特征图的尺寸（Downsampling）和压缩通道数。
* 结构： 通常包含 `BN -> ReLU -> Conv(1x1) -> Average Pooling(2x2)`。
* 压缩系数 (Compression, $\theta$): 为了进一步减少参数，Transition Layer 会通过 $1\times1$ 卷积减少通道数。如果输入通道是 $m$，输出通道通常是 $\lfloor \theta m \rfloor$，其中 $0 < \theta \le 1$。

##### 增长率 (Growth Rate, $k$)

这是 DenseNet 特有的一个超参数。

* 如果每一层 $H_l$ 产生 $k$ 个特征图（feature maps），那么第 $l$ 层的输入通道数就是 $k_0 + k \times (l-1)$（$k_0$ 是输入层的通道数）。
* DenseNet 的一个显著特点是 $k$ 可以取得很小（例如 $k=12$ 或 $k=32$）。这被称为“窄层”（Narrow Layers）。因为网络通过拼接保留了全局状态，每一层只需要学习很少的新特征即可。

##### 代码实现

```python
import torch
from torch import nn
from d2l import torch as d2l

def conv_block(input_channels, num_channels):
    """
    卷积块函数：DenseNet的基本构建单元
    参数:
        input_channels: 输入通道数
        num_channels: 输出通道数
    返回:
        一个包含批量归一化、激活函数和卷积层的序列模块
    """
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),  # 批量归一化层，对输入进行标准化，加速训练并提高稳定性
        nn.ReLU(),  # ReLU激活函数，引入非线性，将负值变为0
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))  # 3x3卷积层，padding=1保持特征图尺寸不变

class DenseBlock(nn.Module):
    """
    稠密块（Dense Block）：DenseNet的核心组件
    每一层的输入都是前面所有层输出的拼接，实现特征重用
    """
    def __init__(self, num_convs, input_channels, num_channels):
        """
        初始化稠密块
        参数:
            num_convs: 该稠密块中卷积层的数量
            input_channels: 输入通道数
            num_channels: 增长率，每个卷积块新增的通道数
        """
        super(DenseBlock, self).__init__()  # 调用父类的初始化方法
        layer = []  # 创建空列表，用于存储所有卷积块
        for i in range(num_convs):  # 循环创建num_convs个卷积块
            # 计算当前卷积块的输入通道数：原始输入 + 前面i个块的输出
            # 每个块输出num_channels个通道，所以前i个块共输出 i * num_channels 个通道
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)  # 将所有卷积块组合成一个序列模块

    def forward(self, X):
        """
        前向传播函数
        参数:
            X: 输入张量
        返回:
            拼接了所有层输出的张量
        """
        for blk in self.net:  # 遍历稠密块中的每个卷积块
            Y = blk(X)  # 将当前输入X通过卷积块得到输出Y
            # 在通道维度（dim=1）上将输入X和输出Y拼接
            # 这是DenseNet的关键：每层的输入包含前面所有层的特征
            X = torch.cat((X, Y), dim=1)
        return X  # 返回最终拼接的特征图

# 测试稠密块的输出形状
blk = DenseBlock(2, 3, 10)  # 创建一个稠密块：2个卷积层，3个输入通道，增长率为10
X = torch.randn(4, 3, 8, 8)  # 创建随机输入张量：批量大小4，3个通道，8x8的特征图
Y = blk(X)  # 通过稠密块进行前向传播
# 输出形状应该是 (4, 3+2*10, 8, 8) = (4, 23, 8, 8)
# 因为输入3个通道 + 第1层输出10个通道 + 第2层输出10个通道 = 23个通道
print(Y.shape)  # 打印输出张量的形状
# torch.Size([4, 23, 8, 8])

def transition_block(input_channels, num_channels):
    """
    过渡层（Transition Layer）：连接两个稠密块
    作用：减少通道数和特征图尺寸，控制模型复杂度
    参数:
        input_channels: 输入通道数
        num_channels: 输出通道数（通常是输入通道数的一半）
    返回:
        包含批量归一化、激活、1x1卷积和平均池化的序列模块
    """
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),  # 批量归一化
        nn.ReLU(),  # ReLU激活函数
        nn.Conv2d(input_channels, num_channels, kernel_size=1),  # 1x1卷积降低通道数
        nn.AvgPool2d(kernel_size=2, stride=2))  # 2x2平均池化，将特征图尺寸减半

# 测试过渡层的输出形状
blk = transition_block(23, 10)  # 创建过渡层：23个输入通道，10个输出通道
# Y的形状是 (4, 23, 8, 8)，经过过渡层后：
# 1x1卷积将通道数从23降到10
# 2x2平均池化将特征图从8x8降到4x4
# 最终输出形状应该是 (4, 10, 4, 4)
print(blk(Y).shape)  # 打印输出张量的形状
# torch.Size([4, 10, 4, 4])

# 构建DenseNet的第一个模块（类似于ResNet的stem）
b1 = nn.Sequential(
    # 7x7大卷积核，步幅为2，padding为3，将1通道(灰度图)转换为64通道
    # 特征图尺寸减半 (96x96 -> 48x48)
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),  # 对64个通道进行批量归一化
    nn.ReLU(),  # ReLU激活函数
    # 3x3最大池化，步幅为2，padding为1
    # 特征图尺寸再次减半 (48x48 -> 24x24)
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 构建DenseNet的主体部分：多个稠密块和过渡层
num_channels, growth_rate = 64, 32  # 初始通道数64，增长率32（每个卷积块新增32个通道）
num_convs_in_dense_blocks = [4, 4, 4, 4]  # 定义4个稠密块，每个块包含4个卷积层
blks = []  # 创建空列表，用于存储所有的稠密块和过渡层

for i, num_convs in enumerate(num_convs_in_dense_blocks):  # 遍历每个稠密块
    # 添加一个稠密块
    # num_convs: 当前块中的卷积层数量
    # num_channels: 当前块的输入通道数
    # growth_rate: 每层新增的通道数
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    
    # 计算稠密块的输出通道数
    # 输出通道数 = 输入通道数 + 卷积层数 × 增长率
    num_channels += num_convs * growth_rate
    
    # 在稠密块之间添加过渡层（最后一个稠密块后不需要过渡层）
    if i != len(num_convs_in_dense_blocks) - 1:
        # 过渡层将通道数减半，控制模型复杂度
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2  # 更新通道数为减半后的值

# 构建完整的DenseNet网络
net = nn.Sequential(
    b1,  # 第一个模块：包含大卷积和最大池化
    *blks,  # 使用*解包，将列表中所有的稠密块和过渡层依次添加
    nn.BatchNorm2d(num_channels),  # 最后一个稠密块后的批量归一化
    nn.ReLU(),  # ReLU激活函数
    nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化，将任意大小的特征图变为1x1
    nn.Flatten(),  # 展平层，将多维张量压缩为一维向量
    nn.Linear(num_channels, 10))  # 全连接层，输出10个类别的分数（对应Fashion-MNIST的10个类别）

# 设置训练参数并开始训练
lr, num_epochs, batch_size = 0.1, 10, 256  # 学习率0.1，训练10个epoch，批量大小256
# 加载Fashion-MNIST数据集，resize=96将图像调整为96x96
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
# 使用d2l提供的训练函数训练模型
# net: 网络模型
# train_iter: 训练数据迭代器
# test_iter: 测试数据迭代器
# num_epochs: 训练轮数
# lr: 学习率
# d2l.try_gpu(): 如果有GPU则使用GPU，否则使用CPU
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

结果如下：

```python
loss 0.140, train acc 0.949, test acc 0.896
578373.9 examples/sec on cuda:0
Best test acc: 0.896
```

![densenet](./src/densenet.png)

### DarkNet-53(yoloV3)

在 YOLOv3（You Only Look Once version 3）中，最为核心的改进之一就是引入了全新的特征提取网络（Backbone），命名为 Darknet-53。

#### 核心设计理念

Darknet-53 的设计主要基于以下三个关键点：

1. 全卷积网络 (Fully Convolutional): 网络中没有使用全连接层（FC），这使得网络可以接受任意尺寸的输入图片。
2. 残差结构 (Residual Blocks): 借鉴了 ResNet 的思想，引入了 Shortcut Connection（跳跃连接）。这解决了深层网络中的梯度消失问题，使得网络可以构建得更深（从 19 层增加到 53 层）。
3. 步长卷积代替池化 (Stride Convolution vs Pooling): 取消了所有的 Max Pooling 池化层，通过卷积层的 `stride=2` 来实现下采样（Downsampling）。这样做的好处是能够更有效地保留图像的细微特征信息。

#### Darknet-53 的具体结构层级

之所以称为 Darknet-53，是因为它包含了 53 个卷积层（52 个卷积层在特征提取部分 + 1 个全连接层用于分类任务，但在 YOLOv3 检测任务中通常去掉全连接层）。

我们可以将网络分为 5 个主要的尺度阶段（Stages），每个阶段通过步长为 2 的卷积层进行下采样。具体如下图所示：

![DarkNet](./src/DarkNet.svg)

#### 关键组件

1. DBL 组件 (Darknet Conv2D_BN_Leaky)
   Darknet-53 中的最小卷积单元不仅仅是一个卷积操作，它通常包含三个部分，在代码中常被称为 DBL 或 CBL：
   1. Conv2d: 卷积层。
   2. Batch Normalization (BN): 批归一化，用于加速收敛并防止过拟合。
   3. Leaky ReLU: 激活函数。与普通 ReLU 不同，它在负数区域给出一个很小的斜率（通常是 0.1），防止神经元“死亡”。
2. 残差模块 (Residual Unit)
   这是 Darknet-53 能够加深网络的关键。一个残差模块包含：
   1. $1 \times 1$ 卷积： 用于压缩通道数（降维），减少参数量和计算量。
   2. $3 \times 3$ 卷积： 用于特征提取和恢复通道数。
   3. Add 操作： 将模块的输入直接加到 $3 \times 3$ 卷积的输出上（Element-wise addition）。
   结构如下:
   $$Input \rightarrow [1 \times 1 \text{ Conv}] \rightarrow [3 \times 3 \text{ Conv}] \rightarrow \text{Add}(Input) \rightarrow Output$$

## 多GPU训练

### 代码实现

```python
%matplotlib inline
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 初始化模型参数
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# 定义模型
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# 交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')

def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params

new_params = get_params(params, d2l.try_gpu(0))
print('b1 权重:', new_params[1])
print('b1 梯度:', new_params[1].grad)

def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)

data = [torch.ones((1,2), device=d2l.try_gpu(i)) for i in range(d2l.num_gpus())]
print("Before allreduce:\n", data[0])
allreduce(data)
print("After allreduce:\n", data[0])

data = torch.arange(20).reshape(4,5)
devices = [torch.device(f'cuda:{i}') for i in range(d2l.num_gpus())]
split = nn.parallel.scatter(data, devices)
print('input data:\n', data)
print('load into: \n', devices)
print('output: \n', split)

#@save
def split_batch(X, y, devices):
    """将X和y拆分到多个设备上"""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))

def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # 在每个GPU上分别计算损失
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]
    for l in ls:  # 反向传播在每个GPU上分别执行
        l.backward()
    # 将每个GPU的所有梯度相加，并将其广播到所有GPU
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce(
                [device_params[c][i].grad for c in range(len(devices))])
    # 在每个GPU上分别更新模型参数
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # 在这里，我们使用全尺寸的小批量

def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # 将模型参数复制到num_gpus个GPU
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # 为单个小批量执行多GPU训练
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
        # 在GPU0上评估模型
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
          f'在{str(devices)}')

train(num_gpus = 1, batch_size = 256, lr = 0.2)
```

### 使用pytorch库实现

```python
import torch
from torch import nn
from d2l import torch as d2l

#@save
def resnet18(num_classes, in_channels=1):
    """稍加修改的ResNet-18模型"""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels))
        return nn.Sequential(*blk)

    # 该模型使用了更小的卷积核、步长和填充，而且删除了最大汇聚层
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(
        64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net

net = resnet18(10)
# 获取GPU列表
devices = d2l.try_all_gpus()
# 我们将在训练代码实现中初始化网络

def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)
    net.apply(init_weights)
    # 在多个GPU上设置模型
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
          f'在{str(devices)}')

train(net, num_gpus=1, batch_size=256, lr=0.1)
```

### 结果(单GPU)

手动实现结果如下：

```python
测试精度：0.83，1.1秒/轮，在[device(type='cuda', index=0)]
```

![dGPU](./src/dGPU1.svg)

pytorch实现结果如下：

```python
测试精度：0.92，16.4秒/轮，在[device(type='cuda', index=0)]
```

![dGPU2](./src/dGPU2.svg)

## 分布式训练

### 数据并行

假设你有一个巨大的数据集（比如ImageNet），我们把它切分成 $N$ 份，分配给 $N$ 个GPU（节点）。

* **模型副本**：每个GPU上都保存一份完全相同的模型参数副本。

* **数据分发**：在每一步迭代中，不同的GPU读取不同的数据小批量（Mini-batch）。

* **独立计算**：每个GPU根据自己读到的数据，独立计算梯度（Gradient）。

这时候问题来了：**每个GPU算出来的梯度是不一样的，怎么更新模型才能保证所有GPU上的模型参数始终保持一致？** 这就是同步SGD要解决的问题。

### 同步SGD

同步SGD的核心逻辑是：**“全员对齐，一起行动”**。它要求在进行下一次迭代前，所有GPU必须完成当前的梯度计算，并算出“平均梯度”来更新参数。

具体步骤如下：

1. 前向与反向传播 (Forward & Backward Pass)：

   所有的GPU（假设有 $k$ 个工作节点）同时开始工作。它们各自从本地数据中取出一个Batch，计算损失函数，并通过反向传播计算出各自的梯度 $g\_i$。

2. 同步与聚合 (Synchronization & Aggregation)：
   这是关键的一步。系统设立一个同步屏障（Barrier）。
   * 计算快的GPU必须**等待**计算慢的GPU。
   * 当所有GPU都完成了梯度计算后，系统会将所有GPU的梯度收集起来，计算平均梯度：
     $$g_{global} = \frac{1}{k} \sum_{i=1}^{k} g_i$$

3. 参数更新 (Parameter Update)：
   每个GPU利用这个相同的 $g\_{global}$ 来更新自己的模型参数 $w$：
   $$w_{t+1} = w_t - \eta \cdot g_{global}$$
   (其中 $\eta$ 是学习率)

4. 广播 (Broadcast)：
   （视架构而定）更新后的参数确保在所有GPU上是完全一致的，然后大家带着完全相同的参数进入下一轮迭代。

## 计算机视觉

### 数据增强

**数据增强 (Data Augmentation)** 的核心目的是通过**对现有训练数据进行一系列随机变换**，人为地扩充数据集的规模和多样性。

简单来说，就是让模型“见多识广”，防止它死记硬背（过拟合）。

#### 常见的图像数据增强方法

在计算机视觉（CV）领域，数据增强应用最为广泛。主要分为两类：

1. 几何变换 (Geometric Transformations)
改变图像的空间结构，但图像内容本身不变。
* **翻转 (Flip)**：水平翻转或垂直翻转。
* **旋转 (Rotation)**：随机旋转一定角度（如 -30° 到 30°）。
* **裁剪 (Crop)**：随机裁剪图片的一部分，或者中心裁剪。
* **缩放 (Resize/Scale)**：改变图片大小或长宽比。
* **平移 (Translation)**：将图像向上下左右移动。

2. 颜色/像素变换 (Color/Pixel Transformations)
改变图像的像素值，不改变形状。
* **色彩抖动 (Color Jitter)**：随机调整亮度、对比度、饱和度和色调。
* **噪声注入 (Noise Injection)**：加入高斯噪声或椒盐噪声，模拟低质量图片。
* **模糊 (Blur)**：使用高斯模糊等平滑图像。
* **擦除 (Cutout/Random Erasing)**：随机在图像上遮挡一块区域（迫使模型利用局部特征识别物体）。

#### 代码实现

引包，打印测试图片：

```python
%matplotlib inline
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

d2l.set_figsize()
img = d2l.Image.open('./cat.webp')
d2l.plt.imshow(img)
```

resize后的图片如下：

![sjzq1](./src/sjzq1.svg)

定义图片增强的应用函数：

```python
def apply(img, aug, num_rows=2, nums_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * nums_cols)]
    d2l.show_images(Y, num_rows, nums_cols, scale=scale)
```

使用torchvision自带的水平翻转：

```python
apply(img, torchvision.transforms.RandomHorizontalFlip())
```

![sjzq2](./src/sjzq2.svg)

垂直翻转：

```python
apply(img, torchvision.transforms.RandomVerticalFlip())
```

![sjzq3](./src/sjzq3.svg)

局部放大缩小提取：

```python
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2)
)
apply(img, shape_aug)
```

![sjzq4](./src/sjzq4.svg)

明暗，色差，对比度：

```python
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
)
apply(img, color_aug)
```

![sjzq5](./src/sjzq5.svg)

组合上面的效果：

```python
augs = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomHorizontalFlip(),
        color_aug,
        shape_aug
    ]
)
apply(img, augs)
```

![sjzq6](./src/sjzq6.svg)

#### 使用数据增强的训练代码

下载数据集并显示前32张：

```python
all_images = torchvision.datasets.CIFAR10(
    train=True, root="../data", download=True
)
d2l.show_images(
    [all_images[i][0] for i in range(32)], 4, 8, scale=0.8
)
```

![sjzq7](./src/sjzq7.svg)

定义数据增强方法：（当前仅仅对于训练数据进行翻转）

```python
train_augs = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor()
    ]
)

test_augs = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor()
    ]
)
```

加载数据集：

```python
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(
        root="../data", train=is_train, transform=augs, download=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=is_train, num_workers=4
    )
    return dataloader
```

定义训练函数：

```python
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """用多GPU进行小批量训练"""
    if isinstance(X, list):
        # 微调BERT中所需
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """用多GPU进行模型训练"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter, device=devices[0])
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

定义训练参数：

```python
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        
net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

训练：

```python
train_with_data_aug(train_augs, test_augs, net)
```

测试结果：

```python
loss 0.215, train acc 0.925, test acc 0.828
2919.0 examples/sec on [device(type='cuda', index=0)]
```

![sjzq8](./src/sjzq8.svg)

给训练数据不添加翻转效果的训练结果：

```python
loss 0.109, train acc 0.961, test acc 0.785
2890.4 examples/sec on [device(type='cuda', index=0)]
```

![sjzq9](./src/sjzq9.svg)

### 微调（迁移学习）

**微调 (Fine-tuning)** 是迁移学习中最核心、最常用的手段。它的基本思想是：**站在巨人的肩膀上**。

与其从零开始训练一个神经网络（随机初始化权重），我们使用在一个大规模数据集（如ImageNet或整个互联网文本）上预训练好的模型作为起点，针对我们的特定任务进行“微小的调整”。

1. 替换输出层 (Replace the Head)
这是微调的第一步，也是必做的一步。
* **原理**：预训练模型（Backbone）通常有一个特定的输出层。例如，ResNet在ImageNet上预训练，输出层有1000个节点（对应1000类）。但你的任务可能只有2类（猫 vs 狗）。
* **操作**：保留前面的所有层（特征提取器），切掉最后的**全连接层（Classifier Head）**，换成一个节点数等于你任务类别数的新层。
* **初始化**：前面的层加载预训练权重，新加的层使用随机初始化。

2. 冻结部分层 (Layer Freezing)
神经网络的不同层学习到的特征是不同的：
* **底层（靠近输入）**：学习通用的低级特征（如边缘、颜色、纹理）。这些特征在任何图像任务中都很通用。
* **高层（靠近输出）**：学习具体的语义特征（如“眼睛”、“车轮”）。这些特征与特定任务强相关。
基于此，我们有不同的冻结策略：
* **Linear Probing (线性探测)**：冻结**所有**骨干网络（Backbone），只训练最后新加的分类层。这适用于数据量极少且任务与预训练任务非常相似的情况。
* **逐步解冻 (Gradual Unfreezing)**：先只训练最后的新层，待收敛后，再解冻最后几个卷积层进行微调，倒数第二组解冻后再微调……直到（可能）解冻所有层。这能防止梯度剧烈波动破坏预训练的底层特征。

3. 差异化学习率 (Differential Learning Rates)
这是一个非常关键的技巧。
* **问题**：新加的层是随机初始化的，需要较大的梯度来快速学习；而预训练的层已经很完美了，只需要微小的改动。如果你用同样的大学习率去更新所有层，预训练好的权重会被破坏（灾难性遗忘）。
* **策略**：
  * **新层（Head）**：使用较大的学习率（例如 $\eta = 1e^{-3}$）。
  * **预训练层（Body）**：使用极小的学习率（例如 $\eta = 1e^{-5}$ 或 $1e^{-6}$）。
  * 甚至可以设置**层级衰减**：层数越深（越靠近底层），学习率越小。

4. 训练策略上的技巧 (Training Tricks)
* **Warm-up (热身)**：在训练刚开始时，先使用极小的学习率训练几个Epoch，然后再慢慢增加到设定的学习率。这是为了让新加的随机初始化层先“稳定”下来，避免一开始的剧烈梯度传到骨干网络。
* **早停 (Early Stopping)**：微调很容易在小数据集上过拟合，所以一旦验证集Loss不再下降，应立即停止训练。

#### 代码实现

导入包：

```python
%matplotlib inline
import os
import torch
import torchvision
from d2l import torch as d2l
from torch import nn
```

下载数据集：

```python
#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')

train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
```

查看数据：

```python
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
```

定义数据增强：

```python
# 使用RGB通道的均值和标准差，以标准化每个通道
# IMGNET数据集的均值和标准差
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])
```

使用IMGNET预训练的resnet18：

```python
pretrained_net = torchvision.models.resnet18(pretrained=True)

print(pretrained_net.fc)
```

其最后的全连接层如下：

```python
Linear(in_features=512, out_features=1000, bias=True)
```

使用预训练模型，修改最后的全连接层：

```python
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)
```

定义训练函数：

```python
# 如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

使用迁移学习训练：

```python
train_fine_tuning(finetune_net, 5e-5)
```

训练结果：

```python
loss 0.163, train acc 0.934, test acc 0.930
557.0 examples/sec on [device(type='cuda', index=0)]
```

![finetuning](./src/finetuning.svg)

如果不使用预训练模型，直接训练：

```python
scrach_net = torchvision.models.resnet18()
scrach_net.fc = nn.Linear(scrach_net.fc.in_features, 2)
train_fine_tuning(scrach_net, 5e-4, param_group=False)
```

结果如下：

```python
loss 0.474, train acc 0.814, test acc 0.835
858.2 examples/sec on [device(type='cuda', index=0)]
```

![nofinetuning](./src/nofinetuning.svg)

### 物体检测

#### 绘制框

引入包：

```python
%matplotlib inline
import torch
from d2l import torch as d2l
```

设置绘制图像大小并显示图片：

```python
# 设置图像显示的大小，使用 d2l 库提供的默认尺寸设置
d2l.set_figsize()

# 读取当前目录下的 catdog.jpg 图片文件
# plt.imread() 会将图片读取为一个数组（numpy array 或类似格式）
img = d2l.plt.imread('./catdog.jpg')

# 使用 matplotlib 显示读取的图片
# imshow() 函数会将图片数组渲染成可视化的图像
d2l.plt.imshow(img)
```

![catdog](./src/catdog.svg)

注意，上面的图片y轴最上面为0；下面是两种转换函数：

```python
def box_corner_to_center(boxes):
    """从(左上, 右下)格式转换为(中心, 宽度, 高度)格式
    
    参数说明：
    boxes: 形状为 (n, 4) 的张量，每行表示一个边界框
           格式为 [x1, y1, x2, y2]，即左上角和右下角坐标
    
    返回值：
    转换后的边界框，格式为 [cx, cy, w, h]
    其中 cx, cy 是中心点坐标，w, h 是宽度和高度
    """
    # 从 boxes 中提取左上角的 x 坐标（第1列）
    x1 = boxes[:, 0]
    # 从 boxes 中提取左上角的 y 坐标（第2列）
    y1 = boxes[:, 1]
    # 从 boxes 中提取右下角的 x 坐标（第3列）
    x2 = boxes[:, 2]
    # 从 boxes 中提取右下角的 y 坐标（第4列）
    y2 = boxes[:, 3]
    
    # 计算中心点的 x 坐标：(左边 + 右边) / 2
    cx = (x1 + x2) / 2
    # 计算中心点的 y 坐标：(上边 + 下边) / 2
    cy = (y1 + y2) / 2
    # 计算宽度：右边 x 坐标 - 左边 x 坐标
    w = x2 - x1
    # 计算高度：下边 y 坐标 - 上边 y 坐标
    h = y2 - y1
    
    # 使用 torch.stack 将四个一维张量堆叠成二维张量
    # axis=-1 表示在最后一个维度上堆叠，得到形状为 (n, 4) 的结果
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """从(中心, 宽度, 高度)格式转换为(左上, 右下)格式
    
    参数说明：
    boxes: 形状为 (n, 4) 的张量，每行表示一个边界框
           格式为 [cx, cy, w, h]，即中心点坐标和宽高
    
    返回值：
    转换后的边界框，格式为 [x1, y1, x2, y2]
    其中 x1, y1 是左上角坐标，x2, y2 是右下角坐标
    """
    # 从 boxes 中提取中心点的 x 坐标（第1列）
    cx = boxes[:, 0]
    # 从 boxes 中提取中心点的 y 坐标（第2列）
    cy = boxes[:, 1]
    # 从 boxes 中提取宽度（第3列）
    w = boxes[:, 2]
    # 从 boxes 中提取高度（第4列）
    h = boxes[:, 3]
    
    # 计算左上角的 x 坐标：中心 x - 宽度的一半
    x1 = cx - 0.5 * w
    # 计算左上角的 y 坐标：中心 y - 高度的一半
    y1 = cy - 0.5 * h
    # 计算右下角的 x 坐标：中心 x + 宽度的一半
    x2 = cx + 0.5 * w
    # 计算右下角的 y 坐标：中心 y + 高度的一半
    y2 = cy + 0.5 * h
    
    # 使用 torch.stack 将四个一维张量堆叠成二维张量
    # axis=-1 表示在最后一个维度上堆叠，得到形状为 (n, 4) 的结果
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

手动验证转换函数是正确的：

```python
# 定义狗的边界框坐标（左上角和右下角）
# 格式：[左上角x, 左上角y, 右下角x, 右下角y]
# 坐标单位是像素，原点在图片左上角
dog_bbox = [60.0, 45.0, 378.0, 516.0]

# 定义猫的边界框坐标（左上角和右下角）
# 格式：[左上角x, 左上角y, 右下角x, 右下角y]
cat_bbox = [400.0, 112.0, 655.0, 493.0]

# 将两个边界框列表转换为 PyTorch 张量
# 转换后的 boxes 形状为 (2, 4)，即2个边界框，每个4个坐标值
boxes = torch.tensor((dog_bbox, cat_bbox))

# 测试格式转换函数的正确性
# 先将角点格式转换为中心格式，再转换回角点格式
# 使用 == 比较转换后的结果是否与原始 boxes 完全相同
# 如果转换函数正确，应该返回全为 True 的张量
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

输出应该为全真数组。下面是将左上右下格式转换为matplotlib格式：

```python
def bbox_to_rect(bbox, color):
    """将边界框(左上, 右下)格式转换为matplotlib格式(左上, 宽度, 高度)
    
    参数说明：
    bbox: 列表或数组，格式为 [x1, y1, x2, y2]
          x1, y1 是左上角坐标，x2, y2 是右下角坐标
    color: 字符串，指定矩形框的颜色，如 'blue', 'red' 等
    
    返回值：
    matplotlib.patches.Rectangle 对象，可以添加到图像上
    """
    # 创建并返回一个 matplotlib 的 Rectangle（矩形）对象
    return d2l.plt.Rectangle(
        # xy 参数：矩形左上角的坐标 (x, y)
        xy=(bbox[0], bbox[1]),
        # width 参数：矩形的宽度 = 右下角x - 左上角x
        width=bbox[2] - bbox[0],
        # height 参数：矩形的高度 = 右下角y - 左上角y
        height=bbox[3] - bbox[1],
        # fill 参数：False 表示不填充矩形内部，只画边框
        fill=False,
        # edgecolor 参数：边框的颜色
        edgecolor=color,
        # linewidth 参数：边框线条的宽度（像素）
        linewidth=2
    )

# 显示原始图片，并将返回的图像对象保存到 fig 变量
# 这样我们就可以在图片上添加其他元素（如边界框）
fig = d2l.plt.imshow(img)

# 在图像上添加狗的边界框（蓝色）
# fig.axes 是图像的坐标轴对象
# add_patch() 方法将矩形框添加到图像上
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))

# 在图像上添加猫的边界框（红色）
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
```

绘制出来的效果如下：

![catdog-k](./src/catdog-k.svg)


#### 数据集

引入包：

```python
%matplotlib inline
import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l
```

定义d2l的数据格式，方便读取或者下载数据集：

```python
# 在 d2l 库的数据中心(DATA_HUB)中注册香蕉检测数据集
# DATA_HUB 是一个字典，存储了各种数据集的下载链接和校验码
d2l.DATA_HUB['banana-detection'] = (
    # 第一个元素：数据集的下载 URL
    # DATA_URL 是 d2l 库中定义的基础数据 URL
    d2l.DATA_URL + 'banana-detection.zip',
    # 第二个元素：数据集的 SHA-1 校验码，用于验证下载文件的完整性
    # 确保下载的文件没有损坏或被篡改
    '5de26c8fce5ccdea9f91267273464dc968d20d72'
)
```

读取数据集：

```python
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签
    
    参数说明：
    is_train: 布尔值，True 表示读取训练集，False 表示读取验证集
    
    返回值：
    images: 图像列表，每个元素是一个图像张量
    targets: 标签张量，形状为 (样本数, 1, 5)，包含类别和边界框坐标
    """
    # 下载并解压香蕉检测数据集，返回数据集所在的目录路径
    # 一般为../data/banana-detection
    data_dir = d2l.download_extract('banana-detection')
    
    # 构造 CSV 标签文件的完整路径
    # os.path.join() 用于拼接路径，适配不同操作系统
    # 根据 is_train 参数选择训练集或验证集的文件夹
    csv_fname = os.path.join(
        data_dir, 
        'bananas_train' if is_train else 'bananas_val',  # 选择训练或验证文件夹
        'label.csv'  # 标签文件名
    )
    
    # 使用 pandas 读取 CSV 文件
    # CSV 文件包含图像文件名和对应的标签信息（类别、边界框坐标）
    csv_data = pd.read_csv(csv_fname)
    
    # 将 'img_name' 列设置为索引，方便后续通过图像名称访问数据
    csv_data = csv_data.set_index('img_name')
    
    # 初始化两个空列表，用于存储图像和标签
    images, targets = [], []
    
    # 遍历 CSV 数据的每一行
    # iterrows() 返回索引（img_name）和该行数据（target）
    for img_name, target in csv_data.iterrows():
        # 读取并添加图像到 images 列表
        images.append(
            # torchvision.io.read_image() 读取图像文件并转换为张量
            torchvision.io.read_image(
                # 构造图像文件的完整路径
                os.path.join(
                    data_dir,  # 数据集根目录
                    'bananas_train' if is_train else 'bananas_val',  # 训练或验证文件夹
                    'images',  # 图像文件夹
                    f'{img_name}'  # 图像文件名（使用 f-string 格式化）
                )
            )
        )
        # 将当前行的标签数据转换为列表并添加到 targets
        # target 包含类别和边界框的坐标信息
        targets.append(list(target))
    
    # 将标签列表转换为 PyTorch 张量
    # unsqueeze(1) 在第二个维度增加一维，从 (n, 5) 变为 (n, 1, 5)
    # 除以 256 是为了将像素坐标归一化到 [0, 1] 范围
    # （假设图像尺寸为 256x256）
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

定义数据集类, 简单起见， 只定义`getitem`, `len`方法：

```python
class BananasDataset(torch.utils.data.Dataset):
    """香蕉检测数据集类
    
    继承自 torch.utils.data.Dataset，这是 PyTorch 中自定义数据集的标准方式
    需要实现三个方法：__init__、__getitem__、__len__
    """
    
    def __init__(self, is_train=True):
        """初始化数据集
        
        参数说明：
        is_train: 布尔值，True 表示加载训练集，False 表示加载验证集
        """
        # 调用 read_data_bananas 函数读取图像和标签
        # self.features 存储所有图像数据
        # self.labels 存储所有标签数据
        self.features, self.labels = read_data_bananas(is_train)
        
        # 打印读取的样本数量，方便用户了解数据集大小
        # str(len(self.features)) 将样本数量转换为字符串
        # 使用三元运算符根据 is_train 选择不同的提示文本
        print('read ' + str(len(self.features)) + 
              (' training examples' if is_train else ' validation examples'))

    def __getitem__(self, idx):
        """获取指定索引的样本
        
        参数说明：
        idx: 整数，样本的索引位置
        
        返回值：
        元组 (图像, 标签)
        """
        # 返回第 idx 个样本
        # self.features[idx].float() 将图像张量转换为浮点型（便于后续计算）
        # self.labels[idx] 返回对应的标签
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        """返回数据集的样本总数
        
        返回值：
        整数，数据集中的样本数量
        """
        # 返回特征列表的长度，即样本总数
        return len(self.features)
```

定义加载数据集到内存的函数，一般情况下不能这么做，因为当前数据集较小，则直接加载到内存中：

```python
def load_data_bananas(batch_size):
    """加载香蕉检测数据集，返回训练和验证数据迭代器
    
    参数说明：
    batch_size: 整数，每个批次包含的样本数量
    
    返回值：
    train_iter: 训练集数据迭代器
    val_iter: 验证集数据迭代器
    """
    # 创建训练集的数据加载器（DataLoader）
    train_iter = torch.utils.data.DataLoader(
        # 第一个参数：BananasDataset 训练集对象
        BananasDataset(is_train=True),
        # 第二个参数：批次大小，决定每次迭代返回多少个样本
        batch_size,
        # shuffle=True：每个 epoch 开始时打乱数据顺序
        # 这有助于提高模型的泛化能力，避免学习到数据的顺序信息
        shuffle=True
    )
    
    # 创建验证集的数据加载器
    val_iter = torch.utils.data.DataLoader(
        # 第一个参数：BananasDataset 验证集对象
        BananasDataset(is_train=False),
        # 第二个参数：批次大小
        batch_size
        # 注意：验证集不需要打乱数据，所以没有 shuffle 参数
    )
    
    # 返回训练集和验证集的迭代器
    return train_iter, val_iter
```

设置一些参数，由于一张图中可能存在多个待检测物体，不应给每个batch给的很大；一般来说是给每个图限制其中最多有多少待检测物体。

```python
# 设置批次大小为 32（每次读取 32 个样本）
batch_size = 32

# 设置图像边长为 256 像素（用于后续坐标还原）
edge_size = 256

# 加载香蕉检测数据集，获取训练集和验证集的迭代器
train_iter, val_iter = load_data_bananas(batch_size)

# 从训练集迭代器中获取第一个批次的数据
# iter(train_iter) 将数据加载器转换为迭代器
# next() 获取迭代器的下一个元素（即第一个批次）
batch = next(iter(train_iter))

# 打印批次数据的形状
# batch[0] 是图像数据，形状为 (batch_size, channels, height, width)
# batch[1] 是标签数据，形状为 (batch_size, 1, 5)
# 5 个值分别是：类别、左上角x、左上角y、右下角x、右下角y（归一化坐标）
print(batch[0].shape, batch[1].shape)
```

注意，这里打印的消息如下：

```python
read 100 validation examples
torch.Size([32, 3, 256, 256]) torch.Size([32, 1, 5])
```

其中第一个tensor的内容是：【batch_size, RGP通道数, 图片长, 图片高】；第二个tensor的内容是：【batch_size, 每张图待检测物体上限, 类别+4个坐标点】

显示10张图片：

```python
# 从批次中取出前 10 张图像进行可视化
# batch[0][0:10] 选择前 10 个样本的图像
# permute(0,2,3,1) 调整张量维度顺序：
#   从 (batch, channels, height, width) 
#   变为 (batch, height, width, channels)
#   这是因为 matplotlib 显示图像需要 (height, width, channels) 格式
# 除以 255 将像素值从 [0, 255] 归一化到 [0, 1] 范围
imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255

# 使用 d2l 库的 show_images 函数显示图像网格
# 参数说明：
#   imgs: 要显示的图像列表
#   2: 显示 2 行
#   5: 显示 5 列（总共 2x5=10 张图像）
#   scale=2: 图像显示的缩放比例
# 返回值 axes 是一个包含所有子图坐标轴对象的列表
axes = d2l.show_images(imgs, 2, 5, scale=2)

# 遍历每个子图和对应的标签，在图像上绘制边界框
# zip(axes, batch[1][0:10]) 将坐标轴和标签配对
for ax, label in zip(axes, batch[1][0:10]):
    # 在当前子图上显示边界框
    d2l.show_bboxes(
        ax,  # 当前子图的坐标轴对象
        # label[0][1:5] 提取边界框坐标（跳过第一个元素，即类别）
        # [1:5] 表示取索引 1 到 4 的元素：左上角x、左上角y、右下角x、右下角y
        # * edge_size 将归一化坐标 [0, 1] 还原为实际像素坐标 [0, 256]
        # 外层的 [] 是因为 show_bboxes 需要一个边界框列表
        [label[0][1:5] * edge_size],
        # colors=['w'] 设置边界框颜色为白色（white）
        colors=['w']
    )
```

![banana-10](./src/banana-10.png)

#### 锚框

锚框是预先在图像上定义好的一系列大小和比例固定的参考框。

在深度学习模型（如 Faster R-CNN, SSD, YOLO v3 等）进行预测时，它并不是凭空去猜目标在哪里，而是基于这些锚框进行两步走：

1. 分类：判断这个锚框里是否有物体，以及是什么物体。
2. 回归：如果框内有物体，微调锚框的边缘（偏移量），使其更精确地贴合目标的真实边界（Ground Truth）。

IoU（交并比）计算的是两个集合（通常是预测框 $A$ 和真实框 $B$）的交集面积与并集面积之比。其计算公式如下：

$$J(A,B)=\frac{\vert A \cap B \vert}{\vert A \cup B \vert}$$

赋予锚框标号：

* 每个锚框是一个训练样本
* 将每个锚框，要么标注背景，要么要么关联一个真实边缘框
* 真实数据集中会有大量锚框，这样会产生大量负类样本

我们需要将每个锚框分配到一个类别（某个物体或背景）以及一个偏移量。这个过程本质上是一个二分图匹配问题。可以参考d2l书籍的571页的例子。视频中描述不是很详细，现做描述如下：

分配算法，匈牙利算法的简化版本：假设有 $n$ 个锚框，$m$ 个真实框，我们会构建一个 $n \times m$ 的矩阵，其中每个元素是锚框 $A_i$ 与真实框 $G_j$ 的 IoU 值。

步骤如下：

1. 寻找全局最大值：在整个矩阵中找到 IoU 最大的单元格（假设是 $A_i$ 和 $G_j$）。
2. 锁定匹配：将 $G_j$ 分配给 $A_i$。此时，$A_i$ 就不再参与其他匹配，$G_j$ 也被“领走”了。
3. 剔除行列：从矩阵中删除第 $i$ 行和第 $j$ 列。
4. 循环往复：重复上述过程，直到所有的真实框 $G$ 都找到了对应的锚框。
5. 处理剩余锚框：
    * 对于那些没被选中的锚框，如果它与某个 $G$ 的 IoU 超过了预设阈值（如 0.5），也可以将其分配给该 $G$。
    * IoU 低于阈值的锚框全部设为 负样本（背景）。

去除重复框算法，极大值抑制（NMS）：

在目标检测的预测阶段（Inference），模型通常会针对同一个物体生成大量重叠的预测框。非极大值抑制（Non-Maximum Suppression, NMS） 的作用就是从这些重叠的框中，“压制”掉那些冗余的，只保留最精准的那一个。

步骤如下：

1. 排序：将所有框按置信度得分从高到低进行降序排列。得分最高的框被认为是最有可能是该物体的。
2. 选择与压制：
    1. 取最高分：从列表中取出得分最高的框（称其为 $A$），将其作为“最终保留框”存入结果列表。
    2. 计算 IoU：将剩余的所有框（$B, C, D...$）分别与这个 $A$ 计算 IoU（交并比）。
    3. 剔除重叠者：如果某个框与 $A$ 的 IoU 超过了预设的阈值（通常设为 0.5），说明这个框很可能是在重复预测同一个物体。直接把这个框从候选中删掉（抑制）。
3. 循环：在剩余的候选框中，重复上述步骤：再次取最高分，剔除高重叠框。直到候选列表变为空。

**锚框生成算法视频中讲的很粗糙，只是给出简单实现，现做说明如下：**

算法：基于缩放比和宽高比的组合：这是最基础的方法，即在特征图的每个像素中心，根据预设的参数生成一组锚框。

核心参数：
* 缩放比 (Scale, $s$)：指锚框相对于原始图像的大小（如 0.1, 0.2 等）。
* 宽高比 (Aspect Ratio, $r$)：指锚框的宽与高的比例（如 1:1[1], 1:2[0.5], 2:1[2]）。

那么锚框的宽高可以直接计算出来，设原始图像的宽为$w$高为$s$, 则：（可能有问题，以代码为主）
* 锚框的宽：$w \cdot \sqrt{s \cdot r}$
* 锚框的高：$\frac{h \cdot s}{\sqrt{r}}$

生成逻辑：

如果设定 $n$ 个缩放比 $[s_1, ..., s_n]$ 和 $m$ 个宽高比 $[r_1, ..., r_m]$，理论上每个像素点会产生 $n \times m$ 个锚框。但为了降低计算量，D2L 和许多算法通常只取包含 $s_1$ 或 $r_1$ 的组合，即：

$$(s_1, r_1), (s_1, r_2), \dots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \dots, (s_n, r_1)$$

这样每个像素点只生成 $n + m - 1$ 个锚框。

代码实现如下：

```python
%matplotlib inline
import torch
from d2l import torch as d2l

# 设置PyTorch打印张量时只显示2位小数，方便查看结果
torch.set_printoptions(2)

def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框
    
    参数:
        data: 输入特征图，shape为(batch_size, channels, height, width)
        sizes: 锚框的尺度列表，如[0.75, 0.5, 0.25]
        ratios: 锚框的宽高比列表，如[1, 2, 0.5]
    返回:
        所有锚框的坐标，shape为(1, 锚框总数, 4)
    """
    # 获取输入特征图的高度和宽度
    in_height, in_width = data.shape[-2:]
    # 获取设备信息（CPU或GPU）、尺度数量、宽高比数量
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    # 计算每个像素位置生成的锚框数量 = n + m - 1（避免重复计算size[0]*ratio[0]）
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    # 将尺度列表转换为张量并放到相应设备上
    size_tensor = torch.tensor(sizes, device=device)
    # 将宽高比列表转换为张量并放到相应设备上
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    # 计算在y轴（高度）上的步长，将像素坐标归一化到[0,1]范围
    steps_h = 1.0 / in_height
    # 计算在x轴（宽度）上的步长，将像素坐标归一化到[0,1]范围
    steps_w = 1.0 / in_width

    # 生成锚框的所有中心点
    # 生成每个像素在高度方向的归一化中心坐标
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    # 生成每个像素在宽度方向的归一化中心坐标
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    # 使用meshgrid生成所有像素位置的网格坐标
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    # 将二维网格展平成一维向量，方便后续处理
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成"boxes_per_pixel"个高和宽，
    # 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
    # 计算锚框的宽度：先根据第一个比例和所有尺度计算，再根据第一个尺度和其他比例计算
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # 乘以高宽比来处理矩形输入图像
    # 计算锚框的高度：同样的组合方式
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # 将宽度和高度除以2来获得半宽和半高，构建相对于中心点的偏移
    # stack成(-w/2, -h/2, w/2, h/2)的形式，然后为每个像素位置重复
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # 每个中心点都将有"boxes_per_pixel"个锚框，
    # 所以生成含所有锚框中心的网格，重复了"boxes_per_pixel"次
    # 将每个像素的中心坐标堆叠4次（对应xmin, ymin, xmax, ymax）
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    # 中心坐标加上相对偏移得到最终的锚框坐标（xmin, ymin, xmax, ymax）
    output = out_grid + anchor_manipulations
    # 在第0维增加batch维度，返回shape为(1, 锚框总数, 4)
    return output.unsqueeze(0)

# 读取猫狗图片
img = d2l.plt.imread('./catdog.jpg')
# 获取图片的高度和宽度（前两个维度）
h, w = img.shape[:2]
# 打印图片尺寸信息
print (f'height: {h}, width: {w}')

# 创建一个随机张量模拟特征图，shape为(batch_size=1, channels=3, height=h, width=w)
X = torch.rand(size=(1, 3, h, w))
# 调用multibox_prior生成锚框
# sizes=[0.75, 0.5, 0.25]: 三种尺度
# ratios=[1, 2, 0.5]: 三种宽高比
# 每个像素会生成 3+3-1=5 个锚框
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25],
                      ratios=[1, 2, 0.5])
# 打印锚框张量的形状
# 输出: (batch_size=1, 锚框数量=h*w*5, 4个坐标值)
print(Y.shape)
# 输出：
# height: 561, width: 728
# torch.Size([1, 2042040, 4])

# 将锚框重新整形为(height, width, 每像素锚框数, 4个坐标)
boxes = Y.reshape(h, w, 5, 4)
# 查看位置(250, 250)处的第1个锚框的坐标
# 输出格式: [xmin, ymin, xmax, ymax]
print(boxes[250, 250, 0, :])
# tensor([0.06, 0.07, 0.63, 0.82])

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """在图像上显示所有边界框
    
    参数:
        axes: matplotlib的坐标轴对象
        bboxes: 边界框列表，每个框为[xmin, ymin, xmax, ymax]
        labels: 每个框的标签文本（可选）
        colors: 每个框的颜色（可选）
    """
    def _make_list(obj, default_values=None):
        """辅助函数：将单个对象转换为列表"""
        # 如果对象为None，使用默认值
        if obj is None:
            obj = default_values
        # 如果不是列表或元组，转换为列表
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    # 将标签转换为列表形式
    labels = _make_list(labels)
    # 将颜色转换为列表，默认使用蓝、绿、红、品红、青色
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    # 遍历所有边界框
    for i, bbox in enumerate(bboxes):
        # 循环使用颜色列表中的颜色
        color = colors[i % len(colors)]
        # 将边界框转换为matplotlib的矩形对象
        rect = d2l.bbox_to_rect(bbox.detach().numpy(), color)
        # 将矩形添加到图像上
        axes.add_patch(rect)
        # 如果提供了标签
        if labels and len(labels) > i:
            # 根据框的颜色选择文本颜色（白底用黑字，其他用白字）
            text_color = 'k' if color == 'w' else 'w'
            # 在矩形的左上角添加文本标签
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

# 设置图形的显示大小
d2l.set_figsize()
# 创建缩放因子，用于将归一化坐标[0,1]还原到像素坐标
# (w, h, w, h)对应(xmin, ymin, xmax, ymax)的缩放
bbox_scale = torch.tensor((w, h, w, h))
# 显示原始图片
fig = d2l.plt.imshow(img)
# 在图片上绘制位置(250, 250)处的5个锚框
# 将归一化坐标乘以缩放因子得到像素坐标
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale, [
    's=0.75, r=1',   # 尺度0.75，宽高比1:1
    's=0.5, r=1',    # 尺度0.5，宽高比1:1
    's=0.25, r=1',   # 尺度0.25，宽高比1:1
    's=0.75, r=2',   # 尺度0.75，宽高比2:1
    's=0.75, r=0.5'  # 尺度0.75，宽高比0.5:1
])
```
![mk-1](./src/mk-1.svg)
```python
def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比（IoU）
    
    参数:
        boxes1: 第一组边界框，shape为(N, 4)，格式为(xmin, ymin, xmax, ymax)
        boxes2: 第二组边界框，shape为(M, 4)，格式为(xmin, ymin, xmax, ymax)
    返回:
        IoU矩阵，shape为(N, M)，每个元素是boxes1[i]和boxes2[j]的IoU
    """
    # 定义Lambda函数计算边界框面积 = 宽度 × 高度
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # boxes1,boxes2,areas1,areas2的形状:
    # boxes1：(boxes1的数量,4),
    # boxes2：(boxes2的数量,4),
    # areas1：(boxes1的数量,),
    # areas2：(boxes2的数量,)
    # 计算第一组所有边界框的面积
    areas1 = box_area(boxes1)
    # 计算第二组所有边界框的面积
    areas2 = box_area(boxes2)
    # inter_upperlefts,inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    # 计算相交区域的左上角坐标：取两个框左上角的最大值
    # boxes1[:, None, :2]添加维度用于广播，shape变为(N, 1, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    # 计算相交区域的右下角坐标：取两个框右下角的最小值
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    # 计算相交区域的宽度和高度，如果不相交则为0（使用clamp限制最小值为0）
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # inter_areas和union_areas的形状:(boxes1的数量,boxes2的数量)
    # 计算相交区域的面积 = 宽 × 高
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    # 计算并集面积 = 面积1 + 面积2 - 交集面积
    union_areas = areas1[:, None] + areas2 - inter_areas
    # 返回IoU = 交集面积 / 并集面积
    return inter_areas / union_areas

#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框
    
    参数:
        ground_truth: 真实边界框，shape为(M, 4)
        anchors: 锚框，shape为(N, 4)
        device: 计算设备
        iou_threshold: IoU阈值，默认0.5
    返回:
        anchors_bbox_map: 锚框到真实框的映射，shape为(N,)，-1表示背景
    """
    # 获取锚框数量和真实边界框数量
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 计算所有锚框和真实框之间的IoU矩阵
    # jaccard[i, j]表示第i个锚框和第j个真实框的IoU值
    jaccard = box_iou(anchors, ground_truth)
    # 创建锚框到真实框的映射张量，初始化为-1（表示背景）
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # 根据阈值，决定是否分配真实边界框
    # 对每个锚框，找到IoU最大的真实框及其IoU值
    max_ious, indices = torch.max(jaccard, dim=1)
    # 找出IoU大于等于阈值的锚框索引
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    # 获取这些锚框对应的最佳真实框索引
    box_j = indices[max_ious >= iou_threshold]
    # 将满足阈值条件的锚框分配给对应的真实框
    anchors_bbox_map[anc_i] = box_j
    # 创建用于标记已处理列的张量（填充-1）
    col_discard = torch.full((num_anchors,), -1)
    # 创建用于标记已处理行的张量（填充-1）
    row_discard = torch.full((num_gt_boxes,), -1)
    # 确保每个真实框至少分配给一个锚框（即使IoU小于阈值）
    for _ in range(num_gt_boxes):
        # 找到IoU矩阵中的最大值索引
        max_idx = torch.argmax(jaccard)
        # 将一维索引转换为二维坐标：真实框索引
        box_idx = (max_idx % num_gt_boxes).long()
        # 将一维索引转换为二维坐标：锚框索引
        anc_idx = (max_idx / num_gt_boxes).long()
        # 将该锚框分配给该真实框
        anchors_bbox_map[anc_idx] = box_idx
        # 将该真实框对应的列设为-1，避免重复分配
        jaccard[:, box_idx] = col_discard
        # 将该锚框对应的行设为-1，避免重复分配
        jaccard[anc_idx, :] = row_discard
    # 返回锚框到真实框的映射
    return anchors_bbox_map

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """计算锚框相对于分配的真实边界框的偏移量（用于训练）
    
    参数:
        anchors: 锚框，格式为(xmin, ymin, xmax, ymax)
        assigned_bb: 分配的真实边界框，格式相同
        eps: 极小值，防止除零或对数运算出错
    返回:
        偏移量，格式为(offset_x, offset_y, offset_w, offset_h)
    """
    # 将锚框从角点格式转换为中心格式(center_x, center_y, width, height)
    c_anc = d2l.box_corner_to_center(anchors)
    # 将真实框从角点格式转换为中心格式
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    # 计算中心点的偏移量：(真实框中心 - 锚框中心) / 锚框尺寸
    # 归一化后的偏移量，使其与锚框大小无关
    offset_xy = (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    # 计算宽高的偏移量：log(真实框尺寸 / 锚框尺寸)
    # 使用对数变换使得尺度变化更加平滑
    offset_wh = torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    # 拼接xy偏移和wh偏移，返回完整的偏移量
    return torch.cat([offset_xy, offset_wh], axis=1)

def multibox_target(anchors, labels):
    """使用真实边界框标记锚框（生成训练目标）
    
    参数:
        anchors: 所有锚框，shape为(1, N, 4)
        labels: 真实标签，shape为(batch_size, M, 5)，每行为(类别, xmin, ymin, xmax, ymax)
    返回:
        bbox_offset: 边界框偏移量，shape为(batch_size, N*4)
        bbox_mask: 掩码，标记哪些锚框需要参与损失计算
        class_labels: 类别标签，shape为(batch_size, N)
    """
    # 获取批次大小，并去除anchors的batch维度
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    # 初始化三个列表，用于存储每个样本的结果
    batch_offset, batch_mask, batch_class_labels = [], [], []
    # 获取设备信息和锚框数量
    device, num_anchors = anchors.device, anchors.shape[0]
    # 遍历批次中的每个样本
    for i in range(batch_size):
        # 获取第i个样本的标签
        label = labels[i, :, :]
        # 将锚框分配给真实边界框
        # label[:, 1:]提取真实框坐标（去掉类别列）
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        # 创建边界框掩码：被分配的锚框（非背景）mask为1，背景为0
        # unsqueeze(-1)增加维度后repeat(1,4)，使每个坐标都有mask
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # 将类标签和分配的边界框坐标初始化为零
        # 初始化所有锚框的类别标签为0（背景类）
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        # 初始化所有锚框的分配边界框坐标为0
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        # 找到所有被分配了真实框的锚框索引（即非背景锚框）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        # 获取这些锚框对应的真实框索引
        bb_idx = anchors_bbox_map[indices_true]
        # 为这些锚框分配类别标签（+1是因为0是背景类）
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        # 为这些锚框分配对应的真实框坐标
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        # 计算锚框相对于真实框的偏移量，并应用掩码（背景锚框的偏移量为0）
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        # 将偏移量展平并添加到批次列表
        batch_offset.append(offset.reshape(-1))
        # 将掩码展平并添加到批次列表
        batch_mask.append(bbox_mask.reshape(-1))
        # 将类别标签添加到批次列表
        batch_class_labels.append(class_labels)
    # 将列表堆叠成张量，shape为(batch_size, ...)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    # 返回偏移量、掩码和类别标签
    return (bbox_offset, bbox_mask, class_labels)

# 定义真实边界框：格式为[类别, xmin, ymin, xmax, ymax]
# 类别0是狗，类别1是猫
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],   # 狗的边界框
                         [1, 0.55, 0.2, 0.9, 0.88]])  # 猫的边界框
# 定义5个手动创建的锚框用于演示
anchors = torch.tensor([[0, 0.1, 0.2, 0.3],      # 锚框0
                    [0.15, 0.2, 0.4, 0.4],    # 锚框1
                    [0.63, 0.05, 0.88, 0.98], # 锚框2
                    [0.66, 0.45, 0.8, 0.8],   # 锚框3
                    [0.57, 0.3, 0.92, 0.9]])  # 锚框4

# 显示图片
fig = d2l.plt.imshow(img)
# 绘制真实边界框（ground_truth[:, 1:]去掉类别列）
# 使用黑色('k')标记，标签为'dog'和'cat'
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
# 绘制锚框，使用默认颜色，标签为锚框编号
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
```
![mk-2](./src/mk-2.svg)
```python
# 调用multibox_target生成训练标签
# unsqueeze(dim=0)为anchors和ground_truth添加batch维度
labels = multibox_target(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))

# 打印类标签（labels[2]）
# 0表示背景，1表示狗，2表示猫
print(labels[2])

# 打印边界框偏移量（labels[0]）
# 每四个值(offset_x, offset_y, offset_w, offset_h)表示一个锚框的偏移
print(labels[0])
# 打印边界框掩码（labels[1]）
# 值为1表示该位置需要参与损失计算，0表示背景不参与
print(labels[1])

# tensor([[0, 1, 2, 0, 2]]) 
# tensor([[-0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00,  1.40e-01,  1.00e+00,
#           5.19e-01,  1.44e+00, -1.20e-01,  2.69e-02,  3.36e-01, -3.13e-01,
#          -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -5.71e-02, -1.00e-01,
#           8.34e-07,  1.25e-01]])
# tensor([[0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1.,
#          1., 1.]])

def offset_inverse(anchors, offset_preds):
    """根据锚框和预测的偏移量反推出预测的边界框（用于推理）
    
    参数:
        anchors: 锚框，格式为(xmin, ymin, xmax, ymax)
        offset_preds: 模型预测的偏移量，格式为(offset_x, offset_y, offset_w, offset_h)
    返回:
        predicted_bbox: 预测的边界框，格式为(xmin, ymin, xmax, ymax)
    """
    # 将锚框从角点格式转换为中心格式
    anc = d2l.box_corner_to_center(anchors)
    # 根据预测的中心偏移量计算预测框的中心坐标
    # 公式：预测中心 = 锚框中心 + (预测偏移 × 锚框尺寸 / 10)
    # 除以10是经验性的缩放因子
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    # 根据预测的尺寸偏移量计算预测框的宽高
    # 公式：预测尺寸 = 锚框尺寸 × exp(预测偏移 / 5)
    # 除以5是经验性的缩放因子
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    # 拼接中心坐标和宽高，得到中心格式的预测框
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    # 将预测框从中心格式转换回角点格式
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    """非极大值抑制（NMS）：去除重叠的冗余边界框
    
    参数:
        boxes: 所有预测边界框，shape为(N, 4)
        scores: 每个框的置信度分数，shape为(N,)
        iou_threshold: IoU阈值，超过此值的框会被抑制
    返回:
        keep: 保留的边界框索引列表
    """
    # 按置信度从高到低对边界框进行排序，B存储排序后的索引
    B = torch.argsort(scores, dim=-1, descending=True)
    # 初始化保留列表，用于存储最终保留的边界框索引
    keep = []
    # 循环直到所有框都被处理
    while B.numel() > 0:
        # 取出当前置信度最高的框的索引
        i = B[0]
        # 将其添加到保留列表
        keep.append(i)
        # 如果只剩一个框，结束循环
        if B.numel() == 1: break
        # 计算当前框与剩余所有框的IoU
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        # 找出IoU小于等于阈值的框的索引（这些框与当前框重叠不大，需要保留）
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        # 更新B，只保留IoU小于阈值的框（+1是因为inds相对于B[1:]）
        B = B[inds + 1]
    # 返回保留的边界框索引张量
    return torch.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框（完整的预测流程）
    
    参数:
        cls_probs: 类别概率，shape为(batch_size, 类别数+1, 锚框数)
        offset_preds: 预测的偏移量，shape为(batch_size, 锚框数*4)
        anchors: 锚框，shape为(1, 锚框数, 4)
        nms_threshold: NMS的IoU阈值
        pos_threshold: 正类的置信度阈值
    返回:
        预测结果，shape为(batch_size, 锚框数, 6)，每行为(类别, 置信度, xmin, ymin, xmax, ymax)
    """
    # 获取设备信息和批次大小
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    # 去除anchors的batch维度
    anchors = anchors.squeeze(0)
    # 获取类别数（不包括背景）和锚框数量
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    # 初始化输出列表
    out = []
    # 遍历批次中的每个样本
    for i in range(batch_size):
        # 获取第i个样本的类别概率和偏移量预测
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        # 对每个锚框，找出最大概率的类别及其置信度
        # cls_prob[1:]排除背景类
        conf, class_id = torch.max(cls_prob[1:], 0)
        # 根据锚框和预测偏移量计算预测的边界框
        predicted_bb = offset_inverse(anchors, offset_pred)
        # 使用NMS去除冗余的边界框，返回保留的索引
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        # 创建所有锚框的索引
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        # 合并保留索引和所有索引
        combined = torch.cat((keep, all_idx))
        # 找出只出现一次的索引（即non_keep）
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        # 将keep和non_keep拼接，保持keep在前
        all_id_sorted = torch.cat((keep, non_keep))
        # 将non_keep的框标记为背景（类别-1）
        class_id[non_keep] = -1
        # 按排序后的索引重新排列类别ID
        class_id = class_id[all_id_sorted]
        # 按排序后的索引重新排列置信度和预测框
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        # 找出置信度低于阈值的框
        below_min_idx = (conf < pos_threshold)
        # 将这些框标记为背景
        class_id[below_min_idx] = -1
        # 将低置信度转换为1-conf（表示是背景的置信度）
        conf[below_min_idx] = 1 - conf[below_min_idx]
        # 拼接类别ID、置信度和预测框坐标，形成最终预测信息
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        # 添加到输出列表
        out.append(pred_info)
    # 堆叠所有样本的结果并返回
    return torch.stack(out)

# 定义4个锚框用于演示NMS
anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92],   # 锚框0
                      [0.08, 0.2, 0.56, 0.95],   # 锚框1
                      [0.15, 0.3, 0.62, 0.91],   # 锚框2
                      [0.55, 0.2, 0.9, 0.88]])   # 锚框3
# 假设模型预测的偏移量都为0（即预测框等于锚框）
offset_preds = torch.tensor([0] * anchors.numel())
# 定义类别概率（3个类别：背景、狗、猫）
cls_probs = torch.tensor([[0] * 4,               # 背景的预测概率（全为0）
                      [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                      [0.1, 0.2, 0.3, 0.9]]) # 猫的预测概率

# 显示图片
fig = d2l.plt.imshow(img)
# 绘制4个锚框及其预测的类别和置信度
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9',  # 锚框0：预测为狗，置信度0.9
             'dog=0.8',  # 锚框1：预测为狗，置信度0.8
             'dog=0.7',  # 锚框2：预测为狗，置信度0.7
             'cat=0.9']) # 锚框3：预测为猫，置信度0.9
```
![mk-3](./src/mk-3.svg)
```python
# 调用multibox_detection进行预测
# unsqueeze(dim=0)为所有输入添加batch维度
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)  # NMS的IoU阈值设为0.5
# 输出结果，每行格式为(类别ID, 置信度, xmin, ymin, xmax, ymax)
# 类别ID为-1表示背景或被NMS抑制的框
print(output)

"""
tensor([[[ 0.00,  0.90,  0.10,  0.08,  0.52,  0.92],
         [ 1.00,  0.90,  0.55,  0.20,  0.90,  0.88],
         [-1.00,  0.80,  0.08,  0.20,  0.56,  0.95],
         [-1.00,  0.70,  0.15,  0.30,  0.62,  0.91]]])
"""

# 显示图片
fig = d2l.plt.imshow(img)
# 遍历第一个样本的所有预测结果
for i in output[0].detach().numpy():
    # 如果类别ID为-1（背景），跳过不绘制
    if i[0] == -1:
        continue
    # 构建标签文本：'dog='或'cat=' + 置信度
    # i[0]为类别索引（0表示狗，1表示猫）
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    # 绘制预测框，i[2:]提取坐标(xmin, ymin, xmax, ymax)
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)
```
![mk-4](./src/mk-4.svg)

注：以上代码在2026年已经几乎不常用了，不必深究。

#### R-CNN 区域卷积神经网络

R-CNN 的核心思想可以概括为：先寻找候选区域，再利用 CNN 提取特征。

##### 算法流程：

1. 生成候选区域 (Region Proposals)： 使用 Selective Search 算法从输入图像中提取约 2000 个可能包含物体的候选框。这些框大小不一。
2. 特征提取：
    * 由于 CNN 的全连接层要求输入尺寸固定，R-CNN 将每个候选框强制缩放（Warping）到统一大小（如 $224 \times 224$）。
    * 将缩放后的图像输入预训练的 CNN（如 AlexNet）提取特征，得到一个固定长度的特征向量。
3. SVM 分类： 将特征向量输入到为每个类别训练的线性 SVM 中，判断该框属于哪个类别（包括背景）。
4. 边框回归 (Bounding Box Regression)： 使用回归器对候选框的位置进行精修，使其更贴合真实物体边界。

由于 R-CNN 太慢，Ross Girshick 推出了 Fast R-CNN。其中的关键改进就是：不再对 2000 个原始候选框运行 CNN，而是只对整张图运行一次 CNN，得到特征图（Feature Map），然后通过 RoI 池化 从特征图中“抠出”对应区域。

##### 运作原理：

假设我们预设的输出尺寸为 $H \times W$（例如 $7 \times 7$）：
1. 映射坐标： 将原图上的候选框坐标映射到特征图（Feature Map）上。
2. 划分子窗口： 将映射后的 RoI 区域划分为 $H \times W$ 个网格（Bins）。如果区域大小为 $h \times w$，那么每个网格的大小约为 $(h/H) \times (w/W)$。
3. 最大池化 (Max Pooling)： 在每个网格内取最大值作为该网格的输出。
4. 拼接结果： 所有网格处理完毕后，就得到了一个固定大小为 $H \times W$ 的特征图。

##### Faster R-CNN：

在 Faster R-CNN 之前，候选区域（Region Proposals）通常由 Selective Search 算法生成。这个算法运行在 CPU 上，处理一张图需要约 2 秒钟，成了整个系统的速度“拖油瓶”。

Faster R-CNN 的伟大之处在于：它取消了外部的候选区域算法，设计了一个 RPN (Region Proposal Network，区域生成网络)，让网络自己去学习“哪里可能有物体”。

RPN工作原理：

1. 特征图输入： RPN 共享了主干网络（Backbone，如 VGG16 或 ResNet）生成的特征图（Feature Map）。
2. 滑动窗口与锚点（Anchors）： 在特征图上的每个像素点，RPN 会预设 $k$ 个不同尺度（Scale）和长宽比（Aspect Ratio）的候选框，这些框被称为 Anchors。
    * 通常使用 3 种尺度和 3 种比例，即每个像素点对应 $k=9$ 个 Anchors。
3. 多任务输出：
    * 分类分支 (Classification)： 判断每个 Anchor 是前景（物体）还是背景。
    * 回归分支 (Regression)： 计算 Anchor 偏离真实物体边界（Ground Truth）的偏移量（$\Delta x, \Delta y, \Delta w, \Delta h$）。

Faster R-CNN完整流程如下：

1. 特征提取： 整张图像输入 CNN，得到共享的特征图。
2. 生成候选框： RPN 在特征图上滑动，筛选出得分较高的候选区域（Proposals）。
3. RoI Pooling： 将这些大小不一的候选区域在特征图上对应的部分切出来，并固定到统一尺寸。
4. 分类与精修： 经过全连接层，最后由两个输出头完成：
    * 具体类别的分类（是猫、是狗、还是车？）。
    * 更精确的边框回归（对 RPN 生成的框进行二次修正）。

##### Mask R-CNN:

Mask R-CNN = Faster R-CNN + Mask 预测分支。它不仅能告诉你图中有什么、在哪里，还能精确到像素级地勾勒出物体的轮廓（即“实例分割”，Instance Segmentation）。

Mask R-CNN 在 Faster R-CNN 的基础上增加了一个并行的分支，用于预测每个感兴趣区域（RoI）内的二值掩码（Binary Mask）。

* 分类分支： 预测物体的类别（是人、是车？）。
* 检测分支： 预测物体的边界框（Bounding Box 回归）。
* Mask 分支： 预测物体的像素级遮罩（使用全卷积网络 FCN 实现）。

一个关键技术改进：RoI Align

这是 Mask R-CNN 最具技术含量的改进。在之前的 RoI Pooling 中，为了将不同大小的框对齐到固定尺寸，需要进行“取整（Quantization）”操作。

* RoI Pooling 的问题： 取整会导致像素级的偏差。对于边界框检测来说，几像素的偏差影响不大；但对于像素级分割来说，几像素的位移会导致 Mask 无法对齐物体边缘。
* RoI Align 的方案：
    1. 取消取整： 保留所有的浮点数坐标。
    2. 双线性插值（Bilinear Interpolation）： 在每个网格单元中设置 4 个采样点，利用插值算法精确计算出这些点在特征图上的值。
    3. 聚合： 对采样点取最大值或平均值。

通过 RoI Align，特征图与原始图像实现了像素级的精确对齐，这是实现高质量分割的前提。

##### 双线性插值

在 RoI Align 中，双线性插值（Bilinear Interpolation） 是核心灵魂。它的作用是：当一个采样点的坐标是浮点数（不在像素中心）时，通过周围四个整数像素点的值，推算出这个点的数值。

在 RoI Pooling 中，如果计算出的坐标是 2.7，算法会直接取整变成 2。这种“四舍五入”会导致特征图上的位置偏移，反映回原图可能就是几十像素的误差。

但是，RoI Align 不取整。 如果坐标是 2.7，它就直接停在 2.7 的位置上。但问题来了：计算机里的图像（特征图）是离散的，只有坐标 (2,2), (2,3) 等整数点有值。2.7 这个位置没有现成的值，怎么办？ 这就需要双线性插值来“算”出来。

双线性插值其实就是在两个方向上分别做线性插值：

假设我们要计算点 $P(x, y)$ 的值，它落在周围四个整数像素点 $Q_{11}, Q_{12}, Q_{21}, Q_{22}$ 之间。

1. 水平方向插值（x 方向）
   我们先在左右两对点之间进行线性插值，找到两个中间点 $R_1$ 和 $R_2$ 的值：
   * 根据 $P$ 的 $x$ 坐标，在 $Q_{11}$ 和 $Q_{21}$ 之间插值得到 $R_1$。
   * 根据 $P$ 的 $x$ 坐标，在 $Q_{12}$ 和 $Q_{22}$ 之间插值得到 $R_2$。
2. 垂直方向插值（y 方向）
   现在我们有了 $R_1$ 和 $R_2$，再根据 $P$ 的 $y$ 坐标，在这两个点之间进行最后一次插值。

插值的计算如下：

设四个点坐标分别为 $(x_1, y_1), (x_1, y_2), (x_2, y_1), (x_2, y_2)$，则 $f(x,y)$ 近似为：

$$f(x,y) \approx \frac{1}{(x_2-x_1)(y_2-y_1)} \sum_{i=1}^2 \sum_{j=1}^2 f(Q_{ij}) \cdot \text{weight}_{ij}$$

#### SSD 单发多框检测

SSD算法主要为了继续加速。SSD 的核心精髓在于：“一步到位”和“多尺度检测”。

##### 核心思想：单次检测 (Single Shot)

与 R-CNN 系列（Two-stage）不同，SSD 属于 One-stage 算法。

* Two-stage (如 Faster R-CNN)： 第一步先找可能哪里有物体（RPN），第二步再看这是什么物体（R-CNN head）。
* One-stage (如 SSD, YOLO)： 只有一步。直接在特征图上进行密集采样，同时预测物体的类别和位置偏移。

##### 两个技术创新

1. 多尺度特征图检测 (Multi-scale Feature Maps)
   * 在 CNN 提取特征的过程中，特征图的尺寸会逐渐变小。
   * 底层特征图（较大）： 感受野小，分辨率高，适合检测小物体。
   * 高层特征图（较小）： 感受野大，语义信息强，适合检测大物体。
   * SSD 直接在 6 个不同尺度的特征图上分别进行预测，从而实现了对不同大小物体的全覆盖。
2. 预设框 (Default Boxes / Priors)
   类似于 Faster R-CNN 的 Anchors。SSD 在每个特征图的每个像素点上，预设了多个不同比例（如 1:1, 1:2, 2:1 等）和大小的框。
   * 网络不需要凭空预测物体的坐标。
   * 网络只需要预测：这个预设框里有没有物体？ 以及 物体真实边界相对于这个预设框的偏移量是多少。

##### SSD 算法完整执行流程

1. 图像预处理 (Input)
   SSD 对输入图像的尺寸有严格要求。最常见的是 SSD300（输入尺寸 $300 \times 300$）和 SSD512。
   * 图像进入网络前会被统一缩放，并进行减均值等归一化处理。
2. 基础网络特征提取 (Backbone)
   SSD 采用传统的 VGG16 作为基础网络（Backbone），但做了两处关键修改：
   * 去头： 删除了 VGG16 原有的全连接层（FC6, FC7），将其转换成卷积层。
   * 空洞卷积： 为了增加感受野同时保持分辨率，将某些卷积层替换为带孔卷积（Atrous/Dilated Convolution）。
3. 多尺度特征层生成 (Extra Feature Layers)
   在 VGG16 之后，SSD 又额外堆叠了数层卷积层，使得特征图的尺寸不断减小（例如从 $38 \times 38$ 降到 $19 \times 19$、$10 \times 10$ 直至 $1 \times 1$）。
   * 关键点： SSD 选择了其中的 6 层 特征图作为预测层。这种“金字塔”结构让它能够兼顾大物体和小物体的检测。
4. 密集预测 (Prediction)
   这是 SSD 最核心的一步。对于选定的 6 个特征图，每一个都会通过一个 $3 \times 3$ 的卷积核进行检测：
   1. 预设框 (Default Boxes)： 在特征图的每个像素点上，生成 $k$ 个不同比例的初始框。
   2. 分类输出： 为每个预设框预测 $C$ 个类别的置信度（$C$ 为类别总数 + 1 个背景类）。
   3. 回归输出： 为每个预设框预测 4 个偏移量（$\Delta x, \Delta y, \Delta w, \Delta h$），用于修正预设框的位置。
   > 计算量估算： 仅 SSD300 一个模型，全图生成的预测框总数就高达 8732 个。这种密集的采样保证了极高的召回率。
5. 后处理与 NMS (Post-processing)
   由于生成了近九千个预测框，会有大量冗余的框重叠在同一个物体上：
   * 置信度过滤： 首先剔除分类得分过低（如低于 0.5）的框。
   * 非极大值抑制 (NMS)： 核心步骤。对于重叠度（IoU）较高的框，只保留得分最高的那一个，删掉其余重复的框。

##### 代码实现(简化后仍复杂)

具体内容请看`ch6-cv/目标识别-SSD.ipynb`

在colab使用A100训练预测结果如下：

```python
class err 3.20e-03, bbox mae 3.06e-03
8872.3 examples/sec on cuda:0
```

![ssd](./src/ssd.jpeg)

对于图片`banana.jpg`的预测结果如下：

![ssd-pre](./src/ssd-pre.jpeg)

#### YOLO(You Only Look Once)

YOLO的核心思想正如其名：只看一眼。在 YOLO 出现之前，目标检测主流算法（如 R-CNN 系列）通常采用“两步走”策略（先找候选框，再做分类）。而 YOLO 直接将目标检测视为一个单一的回归问题，极大地提升了检测速度。

##### 工作原理

YOLO 将输入的图像划分为 $S \times S$ 个网格（Grid Cell）。如果一个目标的中心落在某个网格中，该网格就负责检测该目标。

统一的回归过程,每个网格会预测：

1. 边界框（Bounding Boxes）：包含中心坐标 $(x, y)$ 和宽高 $(w, h)$。
2. 置信度（Confidence Score）：反映框内包含物体的概率以及预测框的准确性。
3. 类别概率（Class Probabilities）：该物体属于特定类别（如猫、狗、车）的概率。

##### 版本

| 版本           | 主要贡献 / 特点                                                                              |
| -------------- | -------------------------------------------------------------------------------------------- |
| v1             | 开创性地将检测转化为回归问题，速度极快，但对小目标检测较弱。                                 |
| v2（YOLO9000） | 引入锚框（Anchor Boxes）与 Batch Normalization，支持层次化标签与联合训练，扩展类别到 ~9000。 |
| v3             | 引入多尺度检测与改进的预测头/特征融合，显著提升小目标检测性能。                              |
| v4 / v5        | 大量工程优化（如 CSP 结构、Mosaic 数据增强、改进的损失与训练策略），成为工业常用版本。       |
| v8 / v10 / v11 | 持续向轻量化、低延迟与更高吞吐优化，出现无锚框（Anchor‑Free）与更快的实时推理设计。          |

##### V3结构

![yolov3](./src/yolov3.svg)

### 语义分割

分辨图片中的多个物体，分类像素归属于不同的目标。

#### 数据集

首先引入包：

```python
%matplotlib inline
import os
import torch
import torchvision
from d2l import torch as d2l
```

下载数据集, 大小约2G：

```python
# ==================== 下载VOC2012语义分割数据集 ====================
# VOC2012是计算机视觉领域常用的语义分割数据集，大小约2GB
# 语义分割：将图像中的每个像素都分类到特定类别（如人、车、猫等）

# 在d2l的数据集Hub中注册VOC2012数据集的下载链接和校验码
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

# 下载并解压数据集到指定目录，返回数据集的路径
# voc_dir 将包含数据集的根目录路径
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
```

定义读取数据集函数，基于voc数据集的特殊格式：

```python
# ==================== 读取VOC数据集的图像和标注 ====================
def read_voc_images(voc_dir, is_train=True):
    """
    读取所有VOC图像及其对应的语义分割标注
    
    参数:
        voc_dir: VOC数据集的根目录路径
        is_train: 布尔值，True读取训练集，False读取验证集
    
    返回:
        features: 原始RGB图像列表
        labels: 对应的语义分割标注图像列表（每个像素用不同颜色表示不同类别）
    """
    # 根据is_train参数确定读取train.txt还是val.txt
    # 这些txt文件包含了图像文件名列表
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    
    # 设置读取模式为RGB彩色图像
    mode = torchvision.io.image.ImageReadMode.RGB
    
    # 打开txt文件并读取所有图像文件名
    with open(txt_fname, 'r') as f:
        images = f.read().split()  # 将文件内容按空白字符分割成列表
    
    # 初始化特征(原图)和标签(分割标注)列表
    features, labels = [], []
    
    # 遍历所有图像文件名
    for i, fname in enumerate(images):
        # 读取原始RGB图像（JPEGImages目录下的.jpg文件）
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        
        # 读取对应的语义分割标注图像（SegmentationClass目录下的.png文件）
        # 标注图像中，不同颜色代表不同的物体类别
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    
    return features, labels

# 读取训练集的图像和标注
train_features, train_labels = read_voc_images(voc_dir, True)
```

显示5张图片看看效果：

```python
# ==================== 可视化原始图像和标注 ====================
# 显示前5张图像及其对应的分割标注

n = 5  # 要显示的图像数量

# 将原始图像和标注图像合并到一个列表中
# train_features[0:n] 是前5张原始图像
# train_labels[0:n] 是前5张标注图像
imgs = train_features[0:n] + train_labels[0:n]

# 调整图像张量的维度顺序
# 从 (C, H, W) 转换为 (H, W, C)，因为显示函数需要这种格式
# C=通道数(RGB=3), H=高度, W=宽度
imgs = [img.permute(1, 2, 0) for img in imgs]

# 显示图像：2行，每行n张
# 第一行显示原始图像，第二行显示对应的分割标注
d2l.show_images(imgs, 2, n)
```

![yyfg-data1](./src/yyfg-data1.png)

手动定义标注的颜色和标签的关系：

```python
# ==================== 定义VOC数据集的类别和颜色映射 ====================

# VOC_COLORMAP: RGB颜色映射表，每种颜色对应一个物体类别
# 例如：[0, 0, 0]是黑色代表背景，[128, 0, 0]是深红色代表飞机
# 这些颜色在标注图像中用来区分不同的物体类别
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

# VOC_CLASSES: 21个类别的名称，与上面的颜色一一对应
# 索引0是背景，索引1是飞机，索引2是自行车，等等
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
```

构建两个互相转换的函数：

```python
# ==================== 构建颜色到类别索引的映射 ====================

def voc_colormap2label():
    """
    构建从RGB颜色值到类别索引的映射
    
    为什么需要这个映射？
    - 标注图像中每个像素都是RGB颜色值（如[128, 0, 0]）
    - 但神经网络需要的是类别索引（如0, 1, 2...）
    - 这个函数创建一个查找表，快速将RGB转换为类别索引
    
    返回:
        colormap2label: 一维张量，长度为256^3（所有可能的RGB组合）
                       索引是RGB值的整数表示，值是对应的类别索引
    """
    # 创建一个大小为256^3的零张量（可以表示所有RGB组合）
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    
    # 遍历所有预定义的颜色
    for i, colormap in enumerate(VOC_COLORMAP):
        # 将RGB三通道值转换为一个唯一的整数索引
        # 公式: (R * 256 + G) * 256 + B
        # 例如：[128, 0, 0] -> (128*256 + 0)*256 + 0 = 8388608
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    
    return colormap2label


def voc_label_indices(colormap, colormap2label):
    """
    将VOC标注图像中的RGB颜色值转换为类别索引
    
    参数:
        colormap: 标注图像张量，形状为 (3, H, W)
        colormap2label: RGB到类别索引的映射表
    
    返回:
        类别索引张量，形状为 (H, W)，每个元素是0-20之间的类别索引
    """
    # 将张量从 (C, H, W) 转换为 (H, W, C)，并转换为numpy数组
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    
    # 将每个像素的RGB值转换为单一整数索引
    # 对于图像中的每个像素，计算其RGB对应的整数值
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    
    # 使用映射表将整数索引转换为类别索引
    return colormap2label[idx]
```

看看转换效果：

```python
# ==================== 测试颜色到类别索引的转换 ====================
# 将第一张标注图像转换为类别索引矩阵
y = voc_label_indices(train_labels[0], voc_colormap2label())

# 打印图像中一小块区域(10x15像素)的类别索引
# 这样可以看到某个区域内各像素的类别编号
print(y[105:115, 125:140])  # 显示从第105-114行，第125-139列的类别索引

# 打印索引1对应的类别名称（应该是'aeroplane'飞机）
print(VOC_CLASSES[1])
```

```plantext
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
aeroplane
```

定义图像数据增强，并显示效果：

```python
# ==================== 随机裁剪数据增强 ====================

def voc_rand_crop(feature, label, height, width):
    """
    对图像和标注进行相同位置的随机裁剪
    
    为什么需要随机裁剪？
    1. 数据增强：增加训练样本的多样性
    2. 统一尺寸：神经网络需要固定大小的输入
    3. 重要：图像和标注必须裁剪相同位置，保证对应关系
    
    参数:
        feature: 原始图像
        label: 标注图像
        height, width: 裁剪后的目标高度和宽度
    
    返回:
        裁剪后的图像和标注
    """
    # 随机获取裁剪区域的参数（左上角坐标、高度、宽度）
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    
    # 对原始图像进行裁剪
    feature = torchvision.transforms.functional.crop(feature, *rect)
    
    # 对标注图像进行相同位置的裁剪（确保像素级对应）
    label = torchvision.transforms.functional.crop(label, *rect)
    
    return feature, label


# ==================== 可视化随机裁剪效果 ====================
imgs = []

# 对同一张图像进行n次随机裁剪，展示数据增强的效果
for _ in range(n):
    # 每次裁剪得到一对(图像, 标注)
    imgs += voc_rand_crop(
        train_features[0], train_labels[0], 200, 300)

# 调整维度以便显示
imgs = [img.permute(1, 2, 0) for img in imgs]

# 显示裁剪结果：第一行显示n个裁剪后的图像，第二行显示对应的标注
# imgs[::2] 是所有偶数索引（原始图像）
# imgs[1::2] 是所有奇数索引（标注图像）
d2l.show_images(imgs[::2] + imgs[1::2], 2, n)
```

![yyfg-data2](./src/yyfg-data2.png)

构建数据集类：

```python
# ==================== 自定义VOC数据集类 ====================

class VOCSegDataset(torch.utils.data.Dataset):
    """
    用于加载VOC语义分割数据集的自定义Dataset类
    
    这个类封装了数据加载、预处理的全部流程：
    1. 读取图像
    2. 过滤太小的图像
    3. 归一化图像
    4. 在训练时随机裁剪
    """

    def __init__(self, is_train, crop_size, voc_dir):
        """
        初始化数据集
        
        参数:
            is_train: 是否为训练集
            crop_size: 裁剪尺寸(height, width)
            voc_dir: VOC数据集目录
        """
        # 图像归一化：使用ImageNet的均值和标准差
        # 这是计算机视觉中的常见做法，有助于模型收敛
        # mean: 每个通道(RGB)的均值
        # std: 每个通道的标准差
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.crop_size = crop_size
        
        # 读取所有图像和标注
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        
        # 过滤掉尺寸小于crop_size的图像，然后归一化
        # self.filter() 只保留足够大的图像
        # self.normalize_image() 对每张图像进行归一化
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        
        # 同样过滤标注图像
        self.labels = self.filter(labels)
        
        # 创建颜色到类别索引的映射表
        self.colormap2label = voc_colormap2label()
        
        # 打印读取的样本数量
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        """
        归一化图像
        
        步骤:
        1. 将像素值从[0, 255]缩放到[0, 1]
        2. 使用ImageNet的均值和标准差进行标准化
        """
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        """
        过滤掉太小的图像
        
        只保留高度>=crop_size[0] 且 宽度>=crop_size[1] 的图像
        因为太小的图像无法裁剪出所需尺寸
        """
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        """
        获取第idx个样本
        
        这是Dataset类必须实现的方法
        每次DataLoader取数据时都会调用这个方法
        
        返回:
            feature: 裁剪并归一化后的图像
            label: 对应的类别索引标注（已转换为类别索引）
        """
        # 随机裁剪图像和标注到指定尺寸
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        
        # 将标注从RGB颜色转换为类别索引
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        """
        返回数据集的大小
        
        这是Dataset类必须实现的方法
        """
        return len(self.features)
```

定义一些训练参数：

```python
# ==================== 创建训练集和测试集 ====================

# 设置裁剪尺寸：高度320像素，宽度480像素
crop_size = (320, 480)

# 创建训练集数据集对象
voc_train = VOCSegDataset(True, crop_size, voc_dir)

# 创建测试集（验证集）数据集对象
voc_test = VOCSegDataset(False, crop_size, voc_dir)

# ==================== 创建数据加载器并测试 ====================

# 设置批次大小：每次训练使用64个样本
batch_size = 64

# 创建数据加载器（DataLoader）
# DataLoader的作用：
# 1. 自动分批次加载数据
# 2. 支持多进程加载，提高效率
# 3. 支持数据打乱（shuffle）
train_iter = torch.utils.data.DataLoader(
    voc_train,                                    # 数据集对象
    batch_size,                                   # 批次大小
    shuffle=True,                                 # 打乱数据顺序（训练时很重要）
    drop_last=True,                               # 丢弃最后不足一个batch的数据
    num_workers=d2l.get_dataloader_workers())     # 多进程加载数据的进程数

# 测试数据加载器：获取一个批次的数据并查看形状
for X, Y in train_iter:
    print(X.shape)  # 图像形状: (batch_size, 3, 320, 480)
                     # 3是RGB三通道
    print(Y.shape)  # 标注形状: (batch_size, 320, 480)
                     # 每个元素是0-20之间的类别索引
    break            # 只看第一个批次
```

封装上述函数：

```python
# ==================== 封装数据加载函数 ====================

def load_data_voc(batch_size, crop_size):
    """
    加载VOC语义分割数据集的便捷函数
    
    这个函数封装了整个数据加载流程，便于在其他地方调用
    
    参数:
        batch_size: 批次大小
        crop_size: 裁剪尺寸(height, width)
    
    返回:
        train_iter: 训练集数据迭代器
        test_iter: 测试集数据迭代器
    """
    # 下载并获取VOC数据集路径
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    
    # 获取合适的worker数量（用于多进程数据加载）
    num_workers = d2l.get_dataloader_workers()
    
    # 创建训练集数据加载器
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir),  # 训练集
        batch_size,
        shuffle=True,                              # 训练时打乱数据
        drop_last=True,                            # 丢弃不完整的批次
        num_workers=num_workers)
    
    # 创建测试集数据加载器
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), # 测试集
        batch_size,
        drop_last=True,                            # 丢弃不完整的批次
        num_workers=num_workers)                   # 测试时不需要打乱
    
    return train_iter, test_iter
```

#### 转置卷积

在卷积神经网络（CNN）的前向传播中，普通的卷积操作通常会降低图像的分辨率（下采样），以提取高层抽象特征。而语义分割要求输出与输入图像尺寸相同的分割掩码（Mask），这就需要上采样（Upsampling）。

转置卷积是一种具有可学习参数的上采样方式。与双线性插值等固定算法不同，转置卷积可以通过反向传播来优化权重，从而学习如何最有效地还原空间细节。

从矩阵乘法的角度来看，普通卷积可以表示为 $\mathbf{y} = \mathbf{C}\mathbf{x}$，其中 $\mathbf{C}$ 是一个稀疏的卷积矩阵。转置卷积的操作则可以看作是 $\mathbf{z} = \mathbf{C}^T\mathbf{y}$。

运作过程（以 Stride=1 为例）：（具体看书）

1. 输入扩大：在输入特征图的像素之间填充（Padding）一定的空位（如果是 Stride > 1，则在像素间插入红点/0）。
2. 卷积运算：使用一个标准的卷积核在扩大后的输入上进行滑动卷积。
3. 输出结果：通过调整卷积核大小、步长（Stride）和填充（Padding），可以将输入特征图“放大”到预期的尺寸。

虽然过程类似于反向的卷积，但不是逆运算，注意，卷积不可逆！

转置卷积的填充不会增大输出，而会减小输出。

一些细节看代码：

```python
# ==================== 导入必要的库 ====================
# torch: PyTorch深度学习框架
import torch

# nn: PyTorch的神经网络模块，包含各种层和损失函数
from torch import nn

# d2l: Dive into Deep Learning工具库
from d2l import torch as d2l

# ==================== 转置卷积的基础实现 ====================

def trans_conv(X, K):
    """
    转置卷积（反卷积）的基础实现
    
    什么是转置卷积？
    - 转置卷积是卷积的逆运算，用于上采样（增大特征图尺寸）
    - 在语义分割中，需要将小的特征图放大到原图大小
    - 转置卷积通过在输入间插入零值，然后进行卷积来实现放大
    
    工作原理：
    - 对输入X的每个元素，与卷积核K相乘
    - 将结果放置在输出的对应位置
    - 有重叠的地方进行累加
    
    参数:
        X: 输入特征图，形状为 (height, width)
        K: 卷积核，形状为 (kernel_h, kernel_w)
    
    返回:
        Y: 输出特征图，形状为 (height+kernel_h-1, width+kernel_w-1)
    
    举例：如果输入是2x2，卷积核是2x2，输出将是3x3
    """
    h, w = K.shape  # 获取卷积核的高度和宽度
    
    # 创建输出张量，尺寸比输入大
    # 输出的高度 = 输入高度 + 卷积核高度 - 1
    # 输出的宽度 = 输入宽度 + 卷积核宽度 - 1
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    
    # 遍历输入的每个元素
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # 将输入元素X[i,j]与整个卷积核相乘
            # 然后将结果加到输出Y的对应区域
            # 这就是为什么叫"转置"卷积：输入的每个值影响输出的一个区域
            Y[i: i + h, j: j + w] += X[i, j] * K
    
    return Y

# ==================== 测试转置卷积 ====================

# 创建一个2x2的输入矩阵
X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])

# 创建一个2x2的卷积核
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])

# 执行转置卷积
# 输入是2x2，卷积核是2x2，输出将是3x3
# 观察输出如何从小的特征图生成大的特征图
trans_conv(X, K)
"""
tensor([[ 0.,  0.,  1.],
        [ 0.,  4.,  6.],
        [ 4., 12.,  9.]])
"""

# ==================== 使用PyTorch内置的转置卷积层 ====================

# 将输入和卷积核reshape为4D张量
# PyTorch的卷积层需要4D输入：(batch_size, channels, height, width)
# reshape(1, 1, 2, 2) 表示：1个样本，1个通道，2x2的特征图
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)

# 创建转置卷积层
# 参数说明：
#   1: 输入通道数
#   1: 输出通道数
#   kernel_size=2: 卷积核大小为2x2
#   bias=False: 不使用偏置项
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)

# 将我们自定义的卷积核赋值给层的权重
# 这样可以验证PyTorch的实现与我们的手动实现是否一致
tconv.weight.data = K

# 执行转置卷积
# 输出应该与我们手动实现的trans_conv(X, K)结果相同
tconv(X)
"""
tensor([[[[4.]]]], grad_fn=<ConvolutionBackward0>)
"""

# ==================== 转置卷积中的填充 ====================

# 创建带填充的转置卷积层
# padding=1: 在输出的四周各去掉1个像素
# 
# 填充的作用：
# - 在普通卷积中，padding增加输入边界
# - 在转置卷积中，padding减少输出尺寸
# - padding=1会从输出的每边去掉1行/列
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K

# 执行带填充的转置卷积
# 对比：不带padding时输出是3x3
#       带padding=1时，输出变为1x1（从3x3的每边去掉1）
tconv(X)
"""
tensor([[[[4.]]]], grad_fn=<ConvolutionBackward0>)
"""

# ==================== 卷积和转置卷积的可逆性验证 ====================

# 创建一个随机输入：1个样本，10个通道，16x16的特征图
X = torch.rand(size=(1, 10, 16, 16))

# 创建普通卷积层
# 10个输入通道 -> 20个输出通道
# kernel_size=5: 5x5的卷积核
# padding=2: 填充2个像素（保持空间尺寸）
# stride=3: 步幅为3（每3个像素移动一次，缩小特征图）
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)

# 创建对应的转置卷积层
# 20个输入通道 -> 10个输出通道（与conv相反）
# 使用相同的kernel_size、padding、stride
# 目的：尝试恢复原始尺寸
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)

# 验证：先卷积再转置卷积，输出形状是否与原始输入相同
# conv(X): 将X下采样
# tconv(conv(X)): 再上采样回去
# 如果设置得当，形状应该恢复到原始大小
# 注意：形状相同不代表值相同，只是尺寸恢复了
tconv(conv(X)).shape == X.shape
# True

# ==================== 卷积与矩阵变换的关系 ====================

# 创建一个3x3的输入矩阵
X = torch.arange(9.0).reshape(3, 3)

# 创建一个2x2的卷积核
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# 使用d2l的corr2d函数执行2D互相关运算（卷积）
# 3x3的输入，2x2的卷积核 -> 2x2的输出
Y = d2l.corr2d(X, K)
print(Y)
"""
tensor([[27., 37.],
        [57., 67.]])

"""

# 这部分展示了卷积操作实际上可以表示为矩阵乘法
# 这对理解转置卷积很重要

# ==================== 将卷积核转换为矩阵形式 ====================

def kernel2matrix(K):
    """
    将2D卷积核转换为矩阵形式
    
    为什么要这样做？
    - 卷积操作可以表示为矩阵乘法
    - 这有助于理解转置卷积的数学原理
    - 如果卷积是 Y = W*X，那么转置卷积就是 Z = W^T*Y
    
    参数:
        K: 2x2的卷积核
    
    返回:
        W: 4x9的权重矩阵，可以通过矩阵乘法实现卷积
    """
    # k: 临时向量，存储展开的卷积核
    # W: 权重矩阵，每一行对应输出的一个元素
    k, W = torch.zeros(5), torch.zeros((4, 9))
    
    # 将2x2卷积核的值填充到长度为5的向量中（中间留一个0）
    # K[0, :] 是卷积核第一行 [1.0, 2.0]
    # K[1, :] 是卷积核第二行 [3.0, 4.0]
    k[:2], k[3:5] = K[0, :], K[1, :]
    
    # 构造4x9的权重矩阵
    # 每一行代表输出的一个位置
    # 每一行包含了该位置对应的卷积核在输入中的布局
    # 这4行对应2x2输出的4个位置
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    
    return W

# 将卷积核转换为矩阵
W = kernel2matrix(K)
print(W)
# W的每一行代表了卷积核在输入上滑动时的一个位置
"""
tensor([[1., 2., 0., 3., 4., 0., 0., 0., 0.],
        [0., 1., 2., 0., 3., 4., 0., 0., 0.],
        [0., 0., 0., 1., 2., 0., 3., 4., 0.],
        [0., 0., 0., 0., 1., 2., 0., 3., 4.]])

"""

# ==================== 验证卷积等价于矩阵乘法 ====================

# 验证：卷积操作 Y = corr2d(X, K) 等价于矩阵乘法 Y = W * X
# 
# 步骤：
# 1. X.reshape(-1): 将3x3的X展平为9x1的向量
# 2. torch.matmul(W, X.reshape(-1)): 4x9矩阵乘以9x1向量 = 4x1向量
# 3. .reshape(2, 2): 将4x1向量重塑为2x2矩阵
# 4. 比较结果是否与Y相同
print(Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2))

# 如果输出全为True，说明卷积确实可以用矩阵乘法表示
# 这是理解转置卷积的关键

# ==================== 验证转置卷积等价于转置矩阵乘法 ====================

# 使用我们实现的trans_conv函数对Y进行转置卷积
# 输入Y是2x2，卷积核K是2x2，输出Z是3x3
Z = trans_conv(Y, K)

# 验证：转置卷积 Z = trans_conv(Y, K) 等价于 Z = W^T * Y
# 
# 关键理解：
# - 如果卷积是 Y = W * X（从大到小）
# - 那么转置卷积就是 Z = W^T * Y（从小到大）
# - W.T 是W的转置矩阵
# 
# 步骤：
# 1. Y.reshape(-1): 将2x2的Y展平为4x1向量
# 2. torch.matmul(W.T, Y.reshape(-1)): 9x4矩阵乘以4x1向量 = 9x1向量
# 3. .reshape(3, 3): 将9x1向量重塑为3x3矩阵
# 4. 比较结果是否与Z相同
print(Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3))

# 如果输出全为True，说明：
# 1. 转置卷积确实是矩阵转置乘法
# 2. 转置卷积可以看作是卷积的"逆操作"（在形状上）
# 3. 这就是为什么它能用于上采样和语义分割
```

#### 全连接卷积神经网络(FCN)

传统的卷积神经网络（CNN）在经过一系列卷积层后，通常会接上几个全连接层（Fully Connected layers）。这会导致两个限制：

* 输入尺寸固定：全连接层要求输入向量的长度必须固定，因此原始图像必须经过裁剪（Crop）或缩放（Resize）。
* 空间信息丢失：全连接层将特征图打平（Flatten），丢失了像素之间的空间相对位置，只能告诉你“图中有一个人”，而不能告诉你“人在哪个位置”。

FCN 的创新之处在于将全连接层全部替换为卷积层。其一般结构为：$Img \rightarrow CNN \rightarrow 1 \times 1 \  Conv \rightarrow Transposed Conv \rightarrow Result$.

参考如下代码，使用预训练模型：

```python
# ==================== 导入必要的库 ====================
# %matplotlib inline: 在Jupyter中内嵌显示matplotlib图像
%matplotlib inline
# torch: PyTorch深度学习框架
import torch
# torchvision: PyTorch的计算机视觉库，包含预训练模型和数据集
import torchvision
# nn: PyTorch的神经网络模块
from torch import nn
# F: PyTorch的函数式接口，包含各种操作函数
from torch.nn import functional as F
# d2l: Dive into Deep Learning工具库
from d2l import torch as d2l

# ==================== 加载预训练的ResNet-18模型 ====================

# 加载在ImageNet上预训练的ResNet-18模型
# 
# 为什么使用预训练模型？
# 1. 迁移学习：利用在大数据集上学到的特征
# 2. 加速训练：不需要从头训练
# 3. 更好的性能：预训练权重提供了良好的初始化
pretrained_net = torchvision.models.resnet18(pretrained=True)

# 查看ResNet-18的最后三层
# children()返回模型的直接子模块列表
# ResNet最后通常是：平均池化层、展平层、全连接层
print(list(pretrained_net.children())[-3:])

"""
[Sequential(
  (0): BasicBlock(
    (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (downsample): Sequential(
      (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (1): BasicBlock(
    (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
), AdaptiveAvgPool2d(output_size=(1, 1)), Linear(in_features=512, out_features=1000, bias=True)]
"""

# ==================== 构建FCN的特征提取部分 ====================

# 去掉ResNet-18的最后两层（全局平均池化和全连接层）
# 
# 为什么要去掉？
# 1. 全连接层会丢失空间信息
# 2. 语义分割需要保留空间位置信息
# 3. FCN的核心思想：全卷积网络，不使用全连接层
# 
# [:-2] 保留除最后两层外的所有层
# *list(...) 将列表展开为参数
# nn.Sequential 将这些层组合成一个顺序模型
net = nn.Sequential(*list(pretrained_net.children())[:-2])

# 测试网络输出形状
# 输入：1张图片，3通道(RGB)，320x480的分辨率
X = torch.randn(1, 3, 320, 480)

# 输出：特征图的形状
# 经过ResNet卷积后，空间尺寸缩小，通道数增加到512
# 预期输出形状：(1, 512, 10, 15)
# 高度：320/32=10，宽度：480/32=15（ResNet下采样32倍）
print(net(X).shape)
# torch.Size([1, 512, 10, 15])

# ==================== 添加FCN的分类和上采样层 ====================

# Pascal VOC数据集包含21个类别
# 0: 背景，1-20: 各种物体类别（飞机、自行车、鸟等）
num_classes = 21

# 添加1x1卷积层：将512个通道转换为21个类别通道
# 
# 1x1卷积的作用：
# 1. 降维：从512通道降到21通道
# 2. 分类：每个通道对应一个类别的预测
# 3. 保持空间信息：不改变特征图的高度和宽度
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))

# 添加转置卷积层：将特征图上采样回原始图像大小
# 
# 参数解释：
# - in_channels=num_classes: 输入21个类别通道
# - out_channels=num_classes: 输出21个类别通道
# - kernel_size=64: 卷积核大小（较大的核用于平滑上采样）
# - padding=16: 填充，用于调整输出尺寸
# - stride=32: 步幅32，将特征图放大32倍（恢复到原始尺寸）
# 
# 为什么stride=32？
# 因为ResNet将图像下采样了32倍，现在需要上采样32倍恢复
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                               kernel_size=64, padding=16,
                                               stride=32))

# ==================== 双线性插值卷积核 ====================

def bilinear_kernel(in_channels, out_channels, kernel_size):
    """
    构造用于双线性插值的转置卷积核
    
    为什么需要双线性插值？
    - 转置卷积的权重需要初始化
    - 双线性插值是一种平滑的上采样方法
    - 它比随机初始化的权重效果更好，训练更稳定
    
    双线性插值原理：
    - 距离中心越近的像素权重越大
    - 距离中心越远的像素权重越小
    - 这样可以产生平滑的放大效果
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
    
    返回:
        weight: 初始化好的卷积核权重
    """
    # 计算中心位置的因子
    # factor用于计算权重衰减的速度
    factor = (kernel_size + 1) // 2
    
    # 确定卷积核的中心位置
    if kernel_size % 2 == 1:
        # 奇数大小：中心是整数位置
        center = factor - 1
    else:
        # 偶数大小：中心是0.5的位置
        center = factor - 0.5
    
    # 创建网格坐标
    # og[0]: 行坐标网格 (kernel_size, 1)
    # og[1]: 列坐标网格 (1, kernel_size)
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    
    # 计算双线性插值的权重
    # 公式：(1 - |x-center|/factor) * (1 - |y-center|/factor)
    # 离中心越近，权重越大；离中心越远，权重越小
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    
    # 初始化权重张量
    # 形状：(in_channels, out_channels, kernel_size, kernel_size)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    
    # 只为对角线位置（输入通道i对应输出通道i）赋值
    # 这保证了每个通道独立处理，不会混合颜色通道
    weight[range(in_channels), range(out_channels), :, :] = filt
    
    return weight

# ==================== 测试双线性插值上采样效果 ====================

# 创建一个转置卷积层用于测试
# 3个输入通道(RGB)，3个输出通道，卷积核4x4
# padding=1, stride=2: 将图像放大2倍
# bias=False: 不使用偏置项
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)

# 用双线性插值核初始化转置卷积的权重
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))

# 读取测试图像并转换为张量
# ToTensor()会将PIL图像转换为形状(C, H, W)的张量，值域[0,1]
img = torchvision.transforms.ToTensor()(d2l.Image.open('./catdog.jpg'))

# 添加批次维度：(C, H, W) -> (1, C, H, W)
X = img.unsqueeze(0)

# 执行转置卷积，放大图像
Y = conv_trans(X)

# 移除批次维度并调整通道顺序以便显示
# (1, C, H, W) -> (C, H, W) -> (H, W, C)
out_img = Y[0].permute(1, 2, 0).detach()

# 设置图像显示大小
d2l.set_figsize()

# 显示原始图像
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0))

# 显示放大后的图像（应该是原始图像的2倍大小）
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img)
# input image shape: torch.Size([561, 728, 3])
# output image shape: torch.Size([1122, 1456, 3])
# <matplotlib.image.AxesImage at 0x7d5bc2196480>
```
![fcn1](./src/fcn1.png)
```python
# ==================== 初始化FCN的转置卷积层 ====================

# 使用双线性插值核初始化FCN网络中的转置卷积层
# 
# 参数说明：
# - num_classes: 21个类别，输入和输出通道都是21
# - 64: 卷积核大小
# 
# 这样初始化的好处：
# 1. 提供了良好的起点，而不是随机权重
# 2. 保证了平滑的上采样效果
# 3. 加速训练收敛
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)

# ==================== 加载VOC数据集 ====================

# 设置批次大小和裁剪尺寸
batch_size = 32           # 每批处理32张图像
crop_size = (320, 480)    # 将图像裁剪为320x480

# 加载VOC语义分割数据集
# train_iter: 训练集数据迭代器
# test_iter: 测试集数据迭代器
# 
# 这个函数会：
# 1. 下载VOC2012数据集（如果还没下载）
# 2. 创建数据集对象
# 3. 创建数据加载器，支持批处理和多进程加载
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)

# ==================== 定义损失函数并开始训练 ====================

def loss(inputs, targets):
    """
    语义分割的损失函数
    
    参数:
        inputs: 模型预测，形状(batch_size, num_classes, H, W)
        targets: 真实标签，形状(batch_size, H, W)，每个元素是类别索引
    
    返回:
        平均损失值
    
    实现细节：
    - cross_entropy: 计算每个像素的交叉熵损失
    - reduction='none': 不自动求平均，保留每个像素的损失
    - .mean(1).mean(1): 对高度和宽度维度求平均
    """
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)


# ==================== 设置训练参数 ====================
num_epochs = 5              # 训练5个epoch
lr = 0.001                  # 学习率
wd = 1e-3                   # 权重衰减（L2正则化）
devices = d2l.try_all_gpus() # 尝试使用所有可用的GPU

# 创建优化器
# SGD: 随机梯度下降
# weight_decay: 权重衰减，防止过拟合
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

# 开始训练
# train_ch13是d2l提供的训练函数，专门用于计算机视觉任务
# 它会：
# 1. 在每个epoch遍历训练数据
# 2. 计算损失并更新权重
# 3. 在测试集上评估性能
# 4. 绘制训练曲线
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

训练结果如下：

```python
loss 0.419, train acc 0.871, test acc 0.849
82.9 examples/sec on [device(type='cuda', index=0)]
```
![fcn2](./src/fcn2.png)

预测：

```python
# ==================== 预测和可视化 ====================

def predict(img):
    """
    对单张图像进行语义分割预测
    
    参数:
        img: 输入图像张量，形状(C, H, W)
    
    返回:
        pred: 预测的类别索引，形状(H, W)
    """
    # 归一化图像（使用与训练时相同的归一化）
    # unsqueeze(0): 添加批次维度 (C, H, W) -> (1, C, H, W)
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    
    # 在GPU上进行预测
    # net(X): 得到形状(1, 21, H, W)的输出，每个通道是一个类别的分数
    # argmax(dim=1): 在类别维度上取最大值的索引，得到(1, H, W)
    pred = net(X.to(devices[0])).argmax(dim=1)
    
    # 移除批次维度，返回(H, W)的类别索引
    return pred.reshape(pred.shape[1], pred.shape[2])


def label2image(pred):
    """
    将类别索引转换为RGB彩色图像
    
    参数:
        pred: 类别索引张量，形状(H, W)
    
    返回:
        彩色分割图，形状(H, W, 3)
    """
    # VOC_COLORMAP是预定义的颜色映射表
    # 将其转换为张量并放到GPU上
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    
    # 确保pred是长整型（用作索引）
    X = pred.long()
    
    # 使用类别索引从颜色映射表中查找对应的RGB颜色
    # colormap[X, :] 会根据X中的类别索引返回对应的RGB值
    return colormap[X, :]


# ==================== 在测试集上进行预测和可视化 ====================

# 下载VOC数据集并获取路径
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')

# 读取测试集的图像和标注
test_images, test_labels = d2l.read_voc_images(voc_dir, False)

# 设置要显示的图像数量
n, imgs = 4, []

# 对前n张测试图像进行预测
for i in range(n):
    # 定义裁剪区域：从左上角(0,0)开始，裁剪320x480的区域
    crop_rect = (0, 0, 320, 480)
    
    # 裁剪原始图像
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    
    # 预测并转换为彩色图像
    pred = label2image(predict(X))
    
    # 收集三张图像：原图、预测结果、真实标注
    imgs += [
        X.permute(1, 2, 0),      # 原始图像，调整为(H,W,C)格式
        pred.cpu(),               # 预测的彩色分割图，移到CPU
        torchvision.transforms.functional.crop(
            test_labels[i], *crop_rect).permute(1, 2, 0)  # 真实标注
    ]

# 显示结果：3行，每行n张图像
# 第一行：原始图像
# 第二行：模型预测结果
# 第三行：真实标注
# imgs[::3]: 每隔3个取一个，即原始图像
# imgs[1::3]: 预测结果
# imgs[2::3]: 真实标注
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
```

预测结果如下：

![fcn3](./src/fcn3.png)

### 样式迁移

在计算机看来，图像只是一堆像素数值。要实现样式迁移，算法必须能够将图像解耦为两个独立的特征维度：

* 内容 (Content): 图像的宏观结构、物体形状和布局（例如：房子在哪里，人站在哪里）。
* 风格 (Style): 图像的纹理、笔触、颜色分布和局部图案（例如：油画的厚重感、素描的线条感）。

深度卷积神经网络 (CNN) 恰好具备这种提取能力。

基于优化的方法 (Gatys et al., 2015)，利用预训练的网络（如 VGG-19）作为特征提取器，通过优化一张随机噪声图像来达到目的。

工作流程：

1. 输入： 准备一张内容图 ($C$) 和一张风格图 ($S$)。
2. 初始化： 生成一张随机噪声图像 ($G$)，或者直接用内容图作为初始 ($G$)。
3. 特征提取： 将这三张图输入到预训练的 VGG 网络中。
4. 计算损失 (Loss)：
   1. 内容损失 ($L_{content}$): 比较 $C$ 和 $G$ 在网络深层（Deep Layers）的特征图。深层特征捕捉的是物体结构，忽略细节。
   2. 风格损失 ($L_{style}$): 比较 $S$ 和 $G$ 在网络多个层（浅层到深层）的特征统计量。
5. 迭代更新： 固定网络参数不变，通过梯度下降改变图像 $G$ 的像素值，使得总损失最小化。

关键数学工具，Gram 矩阵 (Gram Matrix)：

如何用数学描述“风格”？Gatys 引入了 Gram 矩阵。它计算的是特征图（Feature Maps）之间的相关性。如果特征图 A 检测到“垂直线条”，特征图 B 检测到“蓝色”，而它们在 Gram 矩阵中的相关性很高，这就意味着“凡是有垂直线条的地方，往往也是蓝色的”。这种统计规律就是“纹理”或“风格”，而与具体位置无关。

$$L_{total} = \alpha L_{content} + \beta L_{style}$$

其中 $\alpha$ 和 $\beta$ 是超参数，用来控制内容和风格的权重比例。

## RNN

### 序列模型

序列模型的本质就是对“顺序”和“概率”的建模。在传统的机器学习（如逻辑回归、普通的神经网络）中，我们通常假设数据是 i.i.d. (独立同分布) 的。序列模型的核心在于打破了这个假设。它认为数据之间存在前后依赖关系：

1. $x_t$ (当前的事件) 依赖于 $x_{t-1}, x_{t-2}, \dots$ (过去的事件)。
2. 如果不看顺序，数据就失去了意义（例如：“狗咬人”和“人咬狗”，词一样，但顺序变了，意义完全不同）。

序列模型的数学本质：条件概率链：

从数学角度看，序列模型的目标通常是计算一个序列出现的联合概率，或者预测下一个元素的条件概率。假设我们有一个序列 $X = \{x_1, x_2, x_3, \dots, x_T\}$。

序列模型利用概率论中的链式法则 (Chain Rule) 将其分解为：

$$P(X) = P(x_1) \cdot P(x_2 | x_1) \cdot P(x_3 | x_1, x_2) \cdot \dots \cdot P(x_T | x_1, \dots, x_{T-1})$$

序列模型的工作就是拟合这个 $P(x_t | \text{Context})$ 函数。 也就是回答一个问题：“已知前面发生的所有事情，接下来发生这件事的概率是多少？

#### N-Gram 模型 (统计学时代)

这是一个朴素的序列模型。它假设：当前的词只和前面 $N-1$ 个词有关（马尔可夫假设）。

* 原理： 预测“吃”后面是什么，我只看前面2个词“我想”。
* 公式： $P(x_t | x_{t-1}, \dots, x_1) \approx P(x_t | x_{t-1}, \dots, x_{t-n+1})$
* 缺点： “视野”有限。它看不见很久以前的信息，无法处理长距离依赖。

#### 隐马尔可夫模型 (HMM)

它引入了**“隐状态” (Hidden State)** 的概念。

* 原理： 表面看到的序列（观察值）是由背后看不见的序列（隐状态）决定的。
* 缺点： 计算离散状态依然受限，难以通过大规模数据进行复杂的非线性拟合。

### 语言模型

从数学定义的角度看，语言模型是一个概率分布。它的核心任务非常单纯：计算一句话（或一个词序列）在现实世界中出现的概率。或者更直观地说，给定上文，它负责预测下一个词是什么。举个例子：假设由于前面的文本是：“今天天气真好，我想去公园...”

* 模型预测下一个词是 “散步” 的概率可能很高（比如 80%）。
* 预测下一个词是 “游泳” 的概率中等（比如 15%）。
* 预测下一个词是 “吃铁” 的概率几乎为 0（因为这不符合人类语言习惯）。

用数学公式表达，对于一个序列 $w_1, w_2, ..., w_m$，语言模型的目标通常是最大化条件概率：

$$P(w_m | w_1, w_2, ..., w_{m-1})$$

即：在已知前 $m-1$ 个词的情况下，第 $m$ 个词出现的概率。

使用计数来建模，假设序列长度为2, 我们预测：

$$p(x,x') = p(x)p(x'|x)=\frac{n(x)}{n} \cdot \frac{n(x,x')}{n(x)}$$

这里`n`是总词数，`n(x)`,`n(x,x')`是单个单词和连续单词对的出现次数。

### N元语法(n-gram)

假设我们要计算一整句话 $S$ 出现的概率，这句话由词序列 $w_1, w_2, ..., w_m$ 组成。根据概率论中的链式法则（Chain Rule），这句话的真实概率是：
$$P(S) = P(w_1) \times P(w_2 | w_1) \times P(w_3 | w_1, w_2) \times \dots \times P(w_m | w_1, \dots, w_{m-1})$$
但是，当句子很长时，计算 $P(w_m | w_1, \dots, w_{m-1})$ 几乎是不可能的。因为在语料库中，你很难找到一模一样的前 $m-1$ 个词作为历史背景。参数空间太大，数据太稀疏。

基于马尔可夫假设，我们做一个大胆的简化：假设下一个词的出现，只与它前面的 $N-1$ 个词有关，而与更早的词无关。

基于马尔可夫假设，第 $i$ 个词出现的概率近似为：

$$P(w_i | w_1, \dots, w_{i-1}) \approx P(w_i | w_{i-N+1}, \dots, w_{i-1})$$

#### 1元语法

假设： 词与词之间完全独立，不需要看任何历史。

上下文长度： $N-1 = 0$

公式：
$$P(S) \approx P(\text{我}) \times P(\text{爱}) \times P(\text{北京}) \times P(\text{天安门})$$

计算：
$$P(\text{北京}) = \frac{\text{Count(北京)}}{\text{语料库总词数}}$$

特点： 这就是简单的词频统计。它生成的句子通常是一堆杂乱无章的高频词，例如：“的 是 我 了 在”。

#### 2元语法

假设： 当前词只依赖前 1 个词。

上下文长度： $N-1 = 1$。

公式：
$$P(S) \approx P(\text{我} | \text{<s>}) \times P(\text{爱} | \text{我}) \times P(\text{北京} | \text{爱}) \times P(\text{天安门} | \text{北京})$$

计算具体的项 $P(\text{北京} | \text{爱})$：
$$P(\text{北京} | \text{爱}) = \frac{\text{Count}(\text{爱, 北京})}{\text{Count}(\text{爱})}$$
> (翻译：在所有出现“爱”的地方，后面紧跟着“北京”的概率)

#### 三元语法 (Trigram, N=3)

假设： 当前词依赖前 2 个词。

上下文长度： $N-1 = 2$。这是实际应用中（如输入法）最常用的配置之一。

公式：
$$P(S) \approx P(\text{我} | \text{<s>, <s>}) \times \dots \times P(\text{天安门} | \text{我, 爱}) \times P(\text{</s>} | \text{爱, 北京})$$
> (注：通常开头会补两个 \<s\> 以满足 N=3 的窗口)

计算具体的项 $P(\text{天安门} | \text{我, 爱})$：
$$P(\text{天安门} | \text{我, 爱}) = \frac{\text{Count}(\text{我, 爱, 天安门})}{\text{Count}(\text{我, 爱})}$$
> (翻译：在“我爱”这两个词连续出现的情况下，后面接着是“天安门”的概率)

### 循环神经网络

可以将 RNN 想象成一个包含循环的单元。为了更好地理解，我们通常将 RNN 按时间步**“展开”（Unfold）**来看。

工作流程：

1. $t$ 时刻的输入 ($x_t$)： 当前时刻的数据（例如句子中的第 $t$ 个单词）。
2. 前一时刻的隐藏状态 ($h_{t-1}$)： 代表了直到 $t-1$ 时刻的网络记忆。
3. 当前隐藏状态 ($h_t$)： 这是网络的核心计算。它由当前输入和前一记忆共同决定。
   1. 公式： 
   $$h_t = \sigma(W \cdot x_t + U \cdot h_{t-1} + b)$$
   2. 其中，$W$ 和 $U$ 是权重矩阵（在所有时间步共享），$b$ 是偏置，$\sigma$ 是激活函数（通常是 Tanh 或 ReLU）。
4. 输出 ($y_t$)： 基于当前的隐藏状态计算输出。

简单RNN仍然存在梯度消失和梯度爆炸：

梯度消失 (Vanishing Gradient)： 当序列很长时，反向传播算法在计算梯度时，梯度值会随着层数的增加呈指数级衰减。这导致网络无法学到“很久以前”的信息（例如，网络记不住段落开头的名字）。

梯度爆炸 (Exploding Gradient)： 梯度呈指数级增长，导致权重更新过大，网络变得不稳定。

#### 梯度剪裁

给神经网络的参数更新设置了一个“限速器”。如果计算出的梯度太大，就强行把它缩小，以保证模型训练的稳定性。**预防梯度爆炸。**

**按范数裁剪：如果梯度的整体模长超过了阈值，就对其进行等比例缩放。**

公式：设 $g$ 为梯度向量，$C$ 为设定的阈值（例如 1.0 或 5.0），$||g||$ 为梯度的 L2 范数（向量模长，平方和开根）。

$$g_{new} = \left( \frac{C}{\max(||g||, C)} \right) \cdot g$$

   * 如果 $||g|| \le C$：梯度保持不变。
   * 如果 $||g|| > C$：梯度会被缩小，使其模长刚好等于 $C$。

### GRU(门控循环单元)

GRU 旨在解决标准 RNN 中出现的梯度消失 (Vanishing Gradient) 问题，同时保持比 LSTM (长短期记忆网络) 更简单的结构和更少的参数。

GRU 的核心在于通过门控机制 (Gating Mechanism) 来控制信息流。它有两个主要的门：

1. 重置门 (Reset Gate, $r_t$)：决定了如何将新的输入信息与前面的记忆相结合。
2. 更新门 (Update Gate, $z_t$)：决定了保留多少旧的记忆，以及融合多少新的信息。

#### 详细流程和计算公式

假设当前时间步为 $t$，输入为 $x_t$，上一时刻的隐藏状态为 $h_{t-1}$。

1. 计算门控信号
   首先，计算重置门 $r_t$ 和更新门 $z_t$。它们都使用 Sigmoid 函数 ($\sigma$)，将输出压缩到 $[0, 1]$ 之间。
   * 重置门 ($r_t$)：
        $$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$
        > 直观理解：如果 $r_t$ 接近 0，意味着在计算新的候选状态时，模型会“忽略”之前的隐藏状态（相当于重置）。

   * 更新门 ($z_t$)：
        $$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$
        > 直观理解：这个门相当于 LSTM 中“遗忘门”和“输入门”的结合体。它决定了当前时刻的隐藏状态 $h_t$ 中，有多少是直接复制上一时刻的 $h_{t-1}$。
2. 计算候选隐藏状态 ($\tilde{h}_t$)
   这一步计算当前时刻的“新信息”。这里使用了 tanh 激活函数，使数据保持在 $[-1, 1]$ 之间。
   $$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])$$
   (注意：这里用到了重置门 $r_t$。)
   * $\odot$ 表示逐元素相乘 (Hadamard Product)。
   * $r_t \odot h_{t-1}$：如果 $r_t$ 接近 0，模型就会只看当前输入 $x_t$，而忽略过去的 $h_{t-1}$。这允许模型丢弃与未来预测无关的历史信息。
3. 计算最终隐藏状态 ($h_t$)
   最后，模型通过更新门 $z_t$ 来融合“旧记忆”和“新候选状态”。
   $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
   > 直观理解：
   > * 如果 $z_t \approx 1$：模型更多地采纳新的计算结果 $\tilde{h}_t$。
   > * 如果 $z_t \approx 0$：模型更多地保留旧的状态 $h_{t-1}$（这有助于解决梯度消失问题，因为由于 $(1-z_t)$ 的存在，梯度可以直接无损地流向过去）。

两个极端情况：

1. 只有update，没有reset：候选状态永远用完整历史。
2. 只有reset,没有update:所有更新都必须走tanh,梯度指数衰减问题仍然存在。

### LSTM(长短期记忆网络)

在标准 RNN 中，只有一种状态（隐藏状态 $h_t$）在随时间传递，它很容易随着时间推移发生剧烈变化，导致前面的信息被“冲刷”掉。而在 LSTM 中，有两条“线”在并行传递：

1. 隐藏状态 ($h_t$)：类似于短期记忆，用于当前的输出预测，变化较快。
2. 细胞状态 ($C_t$)：类似于长期记忆。它像一条传送带，贯穿所有时间步。信息可以在这条传送带上直线流动，只会有少量的线性交互。这使得由于 Sigmoid 导数引起的梯度衰减问题被大大缓解，信息可以流传得很远。

#### 内部结构

LSTM 通过三个门 (Gates) 来控制向细胞状态中添加信息或删除信息。这三个门分别是：遗忘门、输入门和输出门。假设当前时刻为 $t$，输入为 $x_t$，上一时刻的隐藏状态为 $h_{t-1}$，上一时刻的细胞状态为 $C_{t-1}$。

1. 遗忘门 (Forget Gate) —— “我们要扔掉什么？”
   这是 LSTM 的第一道关卡。它决定了要从上一个细胞状态 $C_{t-1}$ 中丢弃哪些信息。
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
   * 操作：将上一时刻的隐藏状态 $h_{t-1}$ 和当前输入 $x_t$ 拼接，通过 Sigmoid 函数。
   * 结果：输出一个 0 到 1 之间的向量。
     * $1$ 代表“完全保留”。
     * $0$ 代表“完全遗忘”（例如：如果我们在处理文本，看到新的主语是“她”，可能就需要遗忘之前关于主语“他”的性别信息）。
2. 输入门 (Input Gate) —— “我们要存入什么？”
   这一步决定了当前时刻有多少新信息要被存入细胞状态。它分为两个子步骤：
   1. 决定更新哪些值：使用 Sigmoid 层 ($i_t$) 决定更新的力度。
        $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   2. 创建新候选值：使用 Tanh 层创建一个新的候选向量 $\tilde{C}_t$，这是我们想加入的新知识。
        $$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$
3. 更新细胞状态 (Update Cell State) —— “执行更新”
   这是最关键的一步。我们将旧的长期记忆 $C_{t-1}$ 更新为新的 $C_t$。
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
   * $f_t \odot C_{t-1}$：遗忘。旧的记忆乘以遗忘门的输出（如果 $f_t$ 是 0，旧记忆就被擦除）。
   * $i_t \odot \tilde{C}_t$：记忆。新的候选信息乘以输入门的输出（如果 $i_t$ 很大，新信息就被强力写入）。
   * 相加：将保留下来的旧记忆和筛选过的新记忆相加，形成新的长期记忆。注意这里是加法运算，这正是梯度能够长距离传播不消失的数学原因（加法运算的导数为1，梯度可以无损通过）。
4. 输出门 (Output Gate) —— “我们要输出什么？”
   最后，我们需要基于当前的细胞状态 $C_t$ 来决定输出什么（即计算 $h_t$）。
   1. 决定输出部分：使用 Sigmoid 层 ($o_t$) 决定细胞状态的哪些部分需要输出。
        $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   2. 处理细胞状态：将细胞状态 $C_t$ 通过 Tanh 函数（将其值推到 -1 到 1 之间）。
   3. 最终计算：
        $$h_t = o_t \odot \tanh(C_t)$$
        这个 $h_t$ 将作为输出传递给下一层网络，同时也作为下一个时间步的输入。 

### 深度循环神经网络

深度循环神经网络（Deep Recurrent Neural Networks, Deep RNNs）是循环神经网络（RNN）的一种扩展形式。简单来说，它结合了**深度学习（多层结构）和循环神经网络（时间序列处理）**的特性，旨在从序列数据中提取更抽象、更高级的特征。

标准的（浅层）RNN 只有一个隐藏层。虽然它理论上是“图灵完备”的（可以近似任何算法），但在实际应用中，单层网络往往难以捕捉复杂数据中的高阶特征。深度 RNN 的核心思想是在每个时间步（time step）上堆叠多个隐藏层。这意味着信息不仅会在时间轴上（从 $t-1$ 到 $t$）传递，也会在空间轴上（从第 $l$ 层到第 $l+1$ 层）向上传递。

### 双向循环神经网络

双向循环神经网络（Bidirectional Recurrent Neural Networks, 简称 Bi-RNN 或 BRNN）是循环神经网络的一种重要改进架构。在理解序列中的某个点时，不仅要看它的“过去”，还要看它的“未来”。

Bi-RNN 实际上是由两个独立的 RNN 上下堆叠在一起组成的：
1. 前向层（Forward Layer）： 从序列起点到终点（$1 \to T$）处理信息，负责捕捉“上文”特征。记为 $\overrightarrow{h}_t$。
   $$\overrightarrow{h}_t = f(W_{\rightarrow} x_t + U_{\rightarrow} \overrightarrow{h}_{t-1} + b_{\rightarrow})$$
2. 后向层（Backward Layer）： 从序列终点到起点（$T \to 1$）处理信息，负责捕捉“下文”特征。记为 $\overleftarrow{h}_t$。
   $$\overleftarrow{h}_t = f(W_{\leftarrow} x_t + U_{\leftarrow} \overleftarrow{h}_{t+1} + b_{\leftarrow})$$

在每一个时间步 $t$，最终的隐藏状态 $h_t$ 是将前向状态和后向状态进行结合。最常见的结合方式是拼接（Concatenation）。
$$y_t = g(V [\overrightarrow{h}_t, \overleftarrow{h}_t] + b_y)$$

### 编码器-解码器架构

编码器-解码器（Encoder-Decoder） 架构是一种通用的深度学习设计模式，其核心思想是将输入数据“压缩”成一种抽象的中间表示（Feature Representation），然后再将这种表示“解压”或“翻译”成目标输出。

### seq2seq

Seq2Seq（Sequence-to-Sequence） 是一种专门用于处理输入序列和输出序列的模型架构。

它的核心突破在于：输入和输出的序列长度可以是不同且不固定的。 这打破了传统神经网络（如全连接网络）必须输入固定维度、输出固定维度的限制。

Seq2Seq 本质上就是具体的 Encoder-Decoder（编码器-解码器） 架构。标准模型通常由两个 RNN（或 LSTM/GRU）组成。

1. 解码器 (The Decoder)
   * 任务： 负责“读”。它接收输入序列 $X = (x_1, x_2, ..., x_T)$。
   * 工作流： 每一个时间步，RNN 读取一个输入词 $x_t$，并更新自己的隐藏状态 (Hidden State) $h_t$。
        $$h_t = f(h_{t-1}, x_t)$$
   * 最终产出： 编码器读完最后一个词后，产生的最后一个隐藏状态 $h_T$。这个状态理论上包含了整个输入句子的所有信息。
2. 上下文向量 (The Context Vector)
   这是 Seq2Seq 的灵魂:
   * 定义： 通常就是编码器的最后一个隐藏状态 $h_T$（或者是它的变换）。
   * 作用： 它是连接编码器和解码器的唯一桥梁。我们可以把它想象成一个“压缩包”，编码器把整句话的语义压缩成了一个固定长度的向量 $C$。
3. 解码器 (The Decoder)
   任务： 负责“写”。它基于上下文向量 $C$ 生成目标序列 $Y = (y_1, y_2, ..., y_{T'})$。
   工作流：
   1. 初始化： 解码器的初始状态通常设置为上下文向量 $C$。
   2. 逐步生成： 在 $t$ 时刻，解码器根据上一时刻的输出 $y_{t-1}$ 和当前的隐藏状态 $s_t$ 来预测当前词 $y_t$。
   3. 停止： 当解码器输出一个特殊的 <EOS> (End Of Sentence) 标记时，生成结束。

#### BLEU(衡量生成序列好坏)

核心思想是：机器生成的句子（Candidate）和人类专家写的参考句子（Reference）越像，分数就越高。

核心机制：N-gram 精确度 (Precision), 即：BLEU 并不关心语法是否正确，也不关心语义（意思是它读不懂句子），它只关心“词的重合度”。它通过比较 N-gram（连续的 N 个词）来计算。

计算如下：

BLEU 分数通常计算为 1-gram 到 $N$-gram（通常 $N=4$）精确度的几何平均值，再乘以长度惩罚因子。

$$\text{BLEU} = \text{BP} \cdot \exp\left( \sum_{n=1}^{N} w_n \log p_n \right)$$

或者写成几何平均值的形式：

$$\text{BLEU} = \text{BP} \cdot \prod_{n=1}^{N} p_n^{w_n}$$

其中：
* $N$：最大 n-gram 的长度，通常取 4。
* $w_n$：第 $n$ 个 gram 的权重。通常使用均匀权重，即 $w_n = \frac{1}{N}$。
* $p_n$：修正后的 n-gram 精确度。
* $\text{BP}$：简短惩罚因子。

BP 用于惩罚生成的句子长度短于参考句子长度的情况，防止模型生成过短的句子来骗取高精确度。

$$\text{BP} =
\begin{cases}
1 & \text{if } c > r \\
\exp(1 - \frac{r}{c}) & \text{if } c \le r
\end{cases}$$

其中：
* $c$ (candidate length)：机器生成（候选）翻译的长度（单词总数）。
* $r$ (reference length)：参考翻译的有效长度。
* 注：如果有多个参考译文，$r$ 通常取与 $c$ 最接近的那个参考译文的长度。

直观理解：
* 当 $c > r$ 时，指数部分为正，但 BP 被限制最大为 1，不奖励长句。
* 当 $c \le r$ 时，$\frac{r}{c} \ge 1$，导致 $1 - \frac{r}{c} \le 0$。$c$ 越小，指数越负，$\text{BP}$ 值迅速下降（趋近于 0）。

### 束搜索

在深度学习处理序列生成任务（如机器翻译、文本摘要或语音识别）时，RNN（循环神经网络）或其变体（如LSTM、GRU）最后一步往往面临一个抉择：如何从模型输出的词表概率分布中选出最合适的整个句子？束搜索（Beam Search） 正是为此而生的平衡艺术。它是目前自然语言处理领域最常用的解码策略之一。

#### 必要性

在理解束搜索之前，我们需要看看它的两个“极端”替代方案，以此了解它解决了什么痛点：

1. 贪心搜索（Greedy Search）： 每一步都只选当前概率最高的词。优点是计算极快，缺点是“短视”。如果第一步选了一个局部最优但全局次优的词，后面的生成就会步步错，且无法回头（即没有回溯机制）。
2. 穷举搜索（Exhaustive Search）： 尝试词表中的所有可能组合，最后选出整句概率最高的一条。数学上这能保证找到全局最优解，但计算量是指数级的。假设词表大小为 $V$，句子长度为 $T$，时间复杂度是 $O(V^T)$。在实际应用中（词表通常在几万级别），这完全不可行。

束搜索 提供了一个完美的折中方案：它介于极端贪婪与全局搜索之间。它在每一步保留 $k$ 个 累积概率最高的候选序列。这个参数 $k$ 被称为 束宽（Beam Size / Beam Width）。

#### 算法流程

假设我们正在进行机器翻译，设定束宽为 $k=2$（实际应用中通常设为 5 到 10），词表大小为 10000。

1. 初始化：
   模型根据输入句子，预测第一个词的概率分布。我们不只取最高的那一个，而是取概率排名前 2 的词。
   * 候选 A："The" (概率 0.4)
   * 候选 B："A" (概率 0.3)
2. 分支扩张：
   将这 2 个候选词分别作为历史输入，让模型预测下一个词。由于每个候选词都会产生 10000 个新词的概率分布，此时我们共有 $2 \times 10000 = 20000$ 个可能的双词组合。我们计算这 20000 个组合的联合概率，并再次只保留排名前 2 的组合。
   假设排名前 2 的是："The boy" (累积概率 0.15) 和 "A dog" (累积概率 0.10)。
3. 迭代进行：
   重复上述过程。基于保留的 2 个序列，继续预测下一个词，生成 20000 个新组合，从中筛选出累积概率最高的 2 个。
4. 停止条件：
   当某个候选序列生成了代表句子结束的 \<EOS\> (End of Sentence) 标记时，该序列就会停止扩张，被放入一个“完成列表”中。其他没有生成 \<EOS\> 的序列会继续扩张，直到“完成列表”收集到足够数量的完整句子，或者达到了设定的最大生成长度。

#### 数学基础和数值稳定性

序列生成的最终目标是找到一个序列 $y = (y_1, y_2, \dots, y_T)$，使得给定输入 $x$ 的条件概率最大化：

$$P(y|x) = \prod_{t=1}^{T} P(y_t | y_1, \dots, y_{t-1}, x)$$

解决概率下溢（Underflow）：由于单个词的概率都在 0 到 1 之间，将几十个小数相乘，结果会迅速逼近于 0，导致计算机出现浮点数下溢出（Underflow）。为了解决这个问题，束搜索通常在**对数空间（Log Space）**内进行计算。乘法就变成了加法：

$$\log P(y|x) = \sum_{t=1}^{T} \log P(y_t | y_1, \dots, y_{t-1}, x)$$

因为对数函数是单调递增的，所以最大化对数概率等价于最大化原始概率。我们只需在每一步寻找对数概率之和最大的前 $k$ 个序列。

#### 长度偏好

使用对数概率会引入一个新的问题：由于每个词的概率 $P < 1$，所以 $\log P$ 是负数。这意味着，句子越长，累加的负数就越多，总得分就越低。 如果直接使用总对数概率作为评分标准，束搜索会天然地偏向于生成非常短的句子，因为短句子的负项少。

长度惩罚（Length Normalization / Penalty）

为了消除这种偏好，工程上通常会对最终得分进行长度归一化：

$$Score(y) = \frac{1}{T^\alpha} \sum_{t=1}^{T} \log P(y_t | y_1, \dots, y_{t-1}, x)$$

这里的 $T$ 是生成序列的长度，$\alpha$ 是一个调节参数（通常设置在 0.6 到 0.7 之间）。
* 如果 $\alpha = 1$，这就是完全的平均对数概率。
* 如果 $\alpha = 0$，则完全不进行长度归一化。
 
加入长度惩罚后，模型就能公平地评估长句子和短句子了。

## Transformer

### 注意力机制

#### Q、K、V

注意力机制的本质可以理解为一种“软寻址”的信息检索过程。它借鉴了数据库查询的概念，引入了三个核心向量：

* Query (Q) - 查询： 当前正在处理的词（或特征），代表“我想寻找什么信息”。
* Key (K) - 键： 序列中其他词的特征标签，代表“我包含了什么信息”。
* Value (V) - 值： 序列中其他词的实际内容，代表“被检索到的具体信息”。

通俗的比喻： 你去图书馆找书。你的检索词就是 Q，每本书的封面标题和简介就是 K，而书里面的具体内容就是 V。注意力机制计算的就是你的检索词 (Q) 与所有书的标题 (K) 的匹配程度，然后根据匹配程度的高低，为你提取相应的书本内容 (V) 的加权总和。

### NW核回归

这一部分主要为数学推导，主要理解NW核回归事实上是：

$$
f(x) = \sum_{i=1}^{n} \text{Softmax}(x,x_i) \cdot y_i \\
     = \sum_{i=1}^{n} \text{Softmax}(-\frac{1}{2}(x-x_i)^2) \cdot y_i
$$

这里使用了高斯核：$K(\mu) = \frac{1}{\sqrt{2\pi}} \text{exp}(-\frac{{\mu}^2}{2})$

若是带可学习的宽度参数$w$：

$$
f(x) = \sum_{i=1}^{n} \text{Softmax}(-\frac{1}{2}((x-x_i) \cdot w)^2) \cdot y_i
$$

这里的$w$决定了热力图中绘制的权重的宽度。

最后，简单理解NW核回归：区别于平均池化和最大池化的简单`mean`或者`max`，注意力池化将`mean`替换为了一个softmax。

### 注意力评分函数

#### 概念

想象你现在正在图书馆找资料写论文：

* Query (查询 Q)： 你脑子里的问题，比如“如何做红烧肉”。
* Key (键 K)： 图书馆里每本书的书名或标签，比如《川菜大全》、《宇宙简史》、《肉的一百种做法》。
* Value (值 V)： 这些书里的具体内容。

在传统的搜索里，你可能只挑最匹配的一本书看。但在深度学习的“注意力机制”里，模型是贪心的，它想把所有书的内容都看一遍，但根据“书名 (Key)”和“问题 (Query)”的匹配程度，分配不同的阅读精力（权重）。《肉的一百种做法》匹配度极高，分配 80% 的注意力去读里面的内容 (Value)；《川菜大全》匹配度一般，分配 20%；《宇宙简史》毫无关系，分配 0%。最后把读到的内容按照这个比例混合起来，就是最终答案。

注意力评分函数，就是用来计算 Query 和 Key 之间“匹配度”的那个计算器。

#### 计算步骤

计算机只认识数字。所以 Q、K、V 在计算机里都是一长串数字（向量）。打分的过程分为三步：

1. 算原始分数 (Scoring)：这就是注意力评分函数干的活。它把输入的问题 $\mathbf{q}$ 和书签 $\mathbf{k}$ 放进函数里，算出一个原始分数。分数越高，说明越匹配。这个分数可以是任何数值，比如 100，-50，或者 0.5。
2. 处理“凑数的废话” (Masking / 掩蔽)：在处理文本时，句子经常不一样长。为了方便计算，我们会用“空格”或“占位符”（Padding）把短句子补长。但这些占位符是没有意义的！我们绝不能把注意力分给它们。所以，我们要进行**掩蔽 (Masking)**操作：强行把这些占位符的原始匹配分数改成一个极小的负数（比如 $-10^6$）。这样在下一步转换时，它们的注意力就会彻底变成 0。
3. 分数转成百分比 (Softmax)：原始分数有大有小，还可能有负数，没法直接当成“注意力比例”来用。我们需要用一个叫 Softmax 的函数，把所有 Key 的原始分数进行转换。转换后有两个神奇的效果：
   1. 所有的分数都变成了 0 到 1 之间的小数。
   2. 所有的分数加起来刚好等于 1（也就是 100%）。

现在，这个百分比就是最终的注意力权重了。注意力分数是query和key的相似度，注意力权重是分数的softmax结果，两种常见的分数计算：

* q,v长度不同：加性注意力，将q,v合并起来进入一个单输出单隐藏层的MLP。
* q,v长度相同：缩放点积注意力，将q,v做内积。

#### 掩蔽softmax操作

由于输入向量的不同，需要padding占位符保证向量长度相同，但是占位符无意义，在做softmax时不计算这些占位符的比重。通常是设置一个很小的数`1e-6`, 保证指数计算后结果近似为0。

流程：

1. 正常算出所有的原始注意力分数。
2. 找到那些因为“补齐长度”而产生的无效占位符。
3. 把这些占位符的分数强行替换成 $-10^6$。
4. 进行 Softmax 转换，让无效位置的注意力权重彻底变成 0。

#### 加性注意力

在理想状态下，你的问题（Query，简称 Q）和信息源的标签（Key，简称 K）格式是一模一样的。比如你的问题包含 10 个数字（10维向量），标签也包含 10 个数字，那它俩就可以直接“硬碰硬”去计算相似度。但现实往往很骨感。在很多复杂的深度学习任务中，Q 和 K 的长度（维度）是不一样的。

想象一下：
* Query（你）： 是一个只会说中文的人（比如一个 10 维的向量）。
* Key（信息源）： 是一个只会说英语的人（比如一个 20 维的向量）。

由于语言不通、频道不同，你俩没法直接沟通，更别提计算什么“匹配度”了。这时候，我们别无选择，只能花点力气，搭建一个**小型的神经网络（多层感知机）**来当中间人。这就是加性注意力的由来。

工作流程：

为了让不懂中文的 K 和不懂英文的 Q 能够算出匹配分，加性注意力机制组建了一个“翻译兼打分”的裁判团队。这个过程分为四步：

1. 维度对齐：既然 Q 和 K 维度不同，第一步就是把它俩翻译成同一种“世界语”。模型内部准备了两个“翻译矩阵”（数学上叫权重矩阵，$\mathbf{W}_q$ 和 $\mathbf{W}_k$）。
   * $\mathbf{W}_q$ 负责把 Q 翻译成世界语。
   * $\mathbf{W}_k$ 负责把 K 也翻译成世界语。
   经过翻译后，原来长短不一的 Q 和 K，都被统一变成了相同长度的新向量（假设长度为 $h$）。现在，它们终于在同一个频道上了。
2. 加法登场：既然 Q 和 K 已经被翻译成了相同长度的语言，裁判就把它们直接加在一起。这一步的物理意义是：把“你在找什么”和“我这里有什么”这两种信息混合在一起，揉成一个全新的信息团。
3. 激活函数 Tanh：把 Q 和 K 加起来之后，里面的数字可能会变得很大，也可能会变成很小的负数。数字如果失控，后面的计算就容易崩盘。所以，裁判会把这个混合信息团扔进一个叫做 $\tanh$（双曲正切）的函数里。（它像一个温柔的压缩机，不管你扔进去的数字有多大多小，它都会把你平滑地压缩到 -1 到 1 之间）
4. 输出得分：经过前面三步，我们得到了一个规范好的混合向量。但这还是一串数字，而我们需要的最终结果是一个单一的分数（比如 85 分）。这时候，最后一位审判长（一个可学习的权重向量 $\mathbf{w}_v$）出场了。他看了一眼这串混合数字，通过最后一次简单的乘法运算，拍板给出了一个最终的原始匹配分数 $a$。

公式：

$$a(\mathbf{q}, \mathbf{k}) = \mathbf{w}_v^\top \tanh(\mathbf{W}_q\mathbf{q} + \mathbf{W}_k\mathbf{k})$$

其中，
1. $\mathbf{W}_q\mathbf{q}$ 和 $\mathbf{W}_k\mathbf{k}$：两个翻译官把 Q 和 K 变成相同长度。（第一步）
2. $+$：把翻译好的信息加起来。（第二步，这就是“加性”的来源）
3. $\tanh(...)$：用情绪稳定器把数字压缩到 -1 到 1 之间。（第三步）
4. $\mathbf{w}_v^\top$：审判长最后看一眼，算出一个具体的分数。（第四步）

*注：公式里的小 $a$ 就是最后算出来的那个原始分数。这个分数之后还要经过我们上节课讲的 Masked Softmax，才能变成真正的注意力百分比。*

#### 缩放点积注意力

在你的问题（Query，简称 Q）和信息标签（Key，简称 K）的长度（也就是向量的维度 $d$），一模一样的前提下才可使用。在加性注意力里，Q 是 10 维，K 是 20 维也没关系，裁判团队会负责把它们翻译成一样的长度。但在缩放点积注意力中，没有裁判，不包分配。Q 是 512 维，K 就必须也是 512 维。它们必须能直接“对话”。

核心机制：

第一步. 点积：

既然 Q 和 K 长度一样，在数学上，衡量两个向量有多像，最直接、最古老的方法就是算点积。

例：

假设你在找相亲对象，你心里有一个 3 个维度的要求问卷（这就是你的 Query）：

* 要求 1：爱养猫吗？（是为 1，否为 0） -> 你：1
* 要求 2：会写代码吗？ -> 你：1
* 要求 3：喜欢早起吗？ -> 你：0

所以你的 $Q = [1, 1, 0]$。现在有两个候选人（Key）：

* 一号 $K_1$：爱养猫(1)，不会写代码(0)，喜欢早起(1)。$K_1 = [1, 0, 1]$
* 二号 $K_2$：爱养猫(1)，会写代码(1)，不喜欢早起(0)。$K_2 = [1, 1, 0]$

点积是怎么算的呢？就是把对应位置的数字乘起来，然后再把总和加在一起：

* 你和一号的点积：$(1 \times 1) + (1 \times 0) + (0 \times 1) = 1 + 0 + 0 = \mathbf{1分}$
* 你和二号的点积：$(1 \times 1) + (1 \times 1) + (0 \times 0) = 1 + 1 + 0 = \mathbf{2分}$

不需要任何复杂的神经网络或者裁判，两个向量做点积，算出来的数字越大，就说明这俩人越般配（越相似）。

第二步. 缩放：

既然点积这么好用，直接把算出来的分数扔给 Softmax 变成百分比不就行了吗？为什么要加个“缩放（Scaled）”？这是因为会发生“分数爆炸”。刚才我们相亲只有 3 个维度。但在真正的大模型里（比如 GPT），向量的维度 $d$ 动辄几百上千（比如 512 维）。如果有两个 512 维的向量非常匹配，把 512 个乘积加起来，这个点积的原始分数可能会飙升到几百甚至上千。

Softmax 是通过指数函数（$e^x$）来把分数转成百分比的。Softmax 有一个致命弱点：它极度“势利眼”。如果原始分数差距不大（比如 3分 和 5分），它还能按比例分一分（比如 12% 和 88%）。但如果遇到极大值（比如一个 10 分，一个 500 分），Softmax 会直接把 99.9999% 的注意力全给那个 500 分的，其他的全变成 0%。

这会带来一个灾难：赢者通吃，模型变得一根筋。 术语叫做梯度消失——由于只有一个选项拿了满分，其他全是 0，模型在后续的训练中就卡死了，学不到任何新东西了。

既然分数是因为维度 $d$ 太长才加得这么大，那我们只要把算出来的总分，除以维度的平方根（$\sqrt{d}$）就好了。这就像是一个“恒温器”，不管你的向量有多长，除以 $\sqrt{d}$ 后，大家的分数都被重新拉回到一个温和的区间（方差接近 1）。这样 Softmax 就能从容地分配百分比了。

### Bahdanau注意力

在早期的深度学习中，机器翻译（比如把英文翻译成中文）用的是基础的 Seq2Seq 模型。它由两部分组成：

1. 编码器（Encoder）： 负责阅读英文原文。
2. 解码器（Decoder）： 负责写出中文译文。

那时候的工作模式极其反人类，就像是一场闭卷考试：假设你要翻译一句很长的英文：“我昨天去超市买了一个又大又红的苹果，然后回家做了一个苹果派。”编码器（Encoder）会把这句话从头到尾读一遍，然后把所有的信息死记硬背，压缩成一个固定长度的总结向量（叫作上下文变量 $c$）。接着，编码器就被关进小黑屋了。解码器（Decoder）必须仅仅只看这个短短的总结 $c$，把整句中文一点点翻译出来。如果句子太长，那个短小的总结 $c$ 根本装不下那么多细节，解码器翻译到后面就全忘了，导致翻译质量断崖式下跌。

为了拯救记忆力不够的解码器，科学家们把我们之前学的注意力机制装了上去。有了注意力机制后，翻译不再是闭卷考试了，变成了开卷考试！编码器不再只给一个干瘪的总结了，而是把它在阅读每个英文单词时脑子里想的东西（所有的隐藏状态 $\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_T$）全都摊在桌子上，让解码器随时查阅。现在，解码器在每翻译一个中文字时，都可以自由地去原英文句子里“东张西望”，寻找当前最需要的线索。

#### QKV

要理解这个“开卷考试”怎么运作，最关键的是要弄清楚我们在打分时用的 Q（查询）、K（键）、V（值）分别代表什么：

* Query（查询 Q）：解码器当前的状态。
  比如解码器已经翻译出了“我昨天去超市买了一个”，现在正准备翻译下一个词。它脑子里现在的状态就是 Q。它在问：“为了翻译下一个词，我该看英文原文的哪里？”
* Key（键 K）：编码器每个单词的状态。桌子上摊开的英文原文每个词的特征信息。
* Value（值 V）：通常和 Key 是一模一样的，也是编码器每个单词的状态。

#### 举例说明

假设我们要把 "I love apples" 翻译成 "我 爱 苹果"。我们看看解码器准备翻译“爱”这个字时，发生了什么：

1. 提出问题（产生 Query）：解码器刚刚翻译完“我”，当前的状态变成了一个向量 $\mathbf{s}_{t-1}$（这就是此时的 $\mathbf{q}$）。
2. 翻书比对（计算注意力分数）：解码器拿着这个 $\mathbf{q}$，去和桌子上摊开的 "I"、"love"、"apples" 的状态（$\mathbf{k}_1, \mathbf{k}_2, \mathbf{k}_3$）一一进行比对打分。
3. 划重点（Softmax 转换权重）：算出的原始分数经过 Softmax 处理后，变成了百分比（权重）。比如模型发现现在最该关注的是 "love" 这个词：
   * "I" 的权重：5%
   * "love" 的权重：90%
   * "apples" 的权重：5%
4. 提取精华（生成动态上下文 $\mathbf{c}_t$）：解码器根据这些百分比，把 "I"、"love"、"apples" 的信息（Values）混合起来。因为 "love" 占了 90%，所以混合出来的新向量 $\mathbf{c}_t$ 几乎全都是关于 "love" 的意思。这个上下文向量 $\mathbf{c}_t$ 是动态的！ 翻译每个字的时候它都不一样，永远只包含当下最需要的信息。
5. 落笔写字（输出结果）：解码器结合自己当前的状态 $\mathbf{s}_{t-1}$ 和刚刚提取出的精华 $\mathbf{c}_t$，十分确信地输出了中文："爱"。

### 多头注意力

#### 必要性

想象你在阅读这句话：“老板今天狠狠地批评了小明，因为他又迟到了。”

对于“他”这个词，要想彻底理解它，你需要从多个不同的角度去分析：

1. 语法角度（指代关系）： “他”指的是谁？（指向“小明”）
2. 情感角度（因果/情绪）： 为什么被批评？（指向“迟到”，且带有负面色彩）
3. 动作角度（主谓宾）： 谁执行了动作？（“老板”批评，“他”被批评）

如果自注意力机制只有一个“头”（Single-Head Attention），它就像一个只能专注一件事的书呆子。在计算注意力分数时，它可能把 99% 的注意力都放在了找“他 = 小明”这个语法关系上，从而忽略了“迟到”和“批评”的逻辑与情感关系。

#### 核心思想

多头注意力的核心思想非常直观：既然一个头看的不全面，那我们就多安排几个头（Head）一起看！我们可以把“多头”想象成聘请了一个包含 $h$ 个专家的“评审团”（在标准的 Transformer 中，通常有 8 个或 16 个专家）：

* 1号专家（语法大师）： 专门负责找代词指代（他->小明）。
* 2号专家（情感大师）： 专门负责捕捉情绪词汇（狠狠地、批评）。
* 3号专家（逻辑大师）： 专门负责找因果关系（因为->迟到）。
* ……

这 $h$ 个专家同时对同一句话进行自注意力计算。每个人戴着不同颜色的眼镜，看到的信息侧重点完全不同。最后，大家把各自得到的信息汇总在一起，这句话的含义就被全方位、立体地解析出来了。

#### 算法流程

要在数学和代码上实现这个“专家评审团”，其实非常简单，就是在原来的自注意力基础上加了“拆分”和“合并”的动作。

第一步：给每个专家配发专属的“眼镜”（线性投影）

在单头注意力中，我们直接用一套权重矩阵把输入的词变成 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$。而在多头注意力中，为了让 $h$ 个专家看到不同的侧面，我们为每个专家 $i$ 准备了他们专属的、独立的权重矩阵：$\mathbf{W}_i^{(q)}, \mathbf{W}_i^{(k)}, \mathbf{W}_i^{(v)}$。输入的词语通过这些专属矩阵，被“投影”（映射）成了只属于该专家的 $\mathbf{q}_i, \mathbf{k}_i, \mathbf{v}_i$。这就好比给 1号专家戴上了看语法的红蓝眼镜，给 2号专家戴上了看情感的偏振眼镜。

第二步：专家们各自闭门打分（独立计算注意力）

现在，每个专家手里都有了自己那份独特的 $\mathbf{q}_i, \mathbf{k}_i, \mathbf{v}_i$。
接下来，所有的专家互不干扰，同时开始并行计算！ 他们用的计算方法，是**“缩放点积注意力”**：

$$\text{head}_i = \text{Attention}(\mathbf{q}\mathbf{W}_i^{(q)}, \mathbf{k}\mathbf{W}_i^{(k)}, \mathbf{v}\mathbf{W}_i^{(v)}) = \text{softmax}\left(\frac{\mathbf{q}_i \mathbf{k}_i^\top}{\sqrt{d_k}}\right)\mathbf{v}_i$$

这就相当于 8 个专家同时交出了 8 份不同视角的“单词理解报告”（$\text{head}_1$ 到 $\text{head}_8$）。

第三步：评审团圆桌会议（拼接与最终融合）

8 份报告交上来了，但计算机最终只需要一个结果。怎么办？

1. 拼接（Concatenate）： 简单粗暴，直接把这 8 份报告按顺序粘在一起，拼成一个超级长的大向量。
2. 融合（最终的线性投影）： 这个超级长的向量太长了，里面有些信息可能有重复。所以，我们最后再请出一位“主编”（一个额外的输出权重矩阵 $\mathbf{W}^{(o)}$）。主编把拼接好的大向量重新融合、提炼，输出最终的、包含所有专家智慧的词向量！
   $$\text{MultiHead} = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}^{(o)}$$

### 自注意力机制

#### 必要性（解决指代问题）

想象有一句英文：

"The animal didn't cross the street because it was too tired."
（这只动物没有过马路，因为它太累了。）

作为人类，你一眼就能看出这里的 "it" (它) 指的是 "animal" (动物)，而不是 "street" (马路)。因为你知道“累”的只能是动物，不能是马路。但是计算机不知道啊！以前用传统的 RNN（循环神经网络）从左到右读句子，读到 "it" 的时候，它可能早就把前面的 "animal" 忘光了，或者搞不清 "it" 到底和前面哪个词最相关。

自注意力机制（Self-Attention）就是为了解决这个“语境和关系”问题而诞生的。 它让句子中的每一个词，都去和其他所有的词相亲（打分），看看谁和自己最般配，从而深刻理解自己在当前句子里的真正含义。

#### QKV

在 Seq2Seq 里，Q 来自解码器，K 和 V 来自编码器。但在自注意力机制中，Q、K、V 全部来自同一个地方——也就是输入的句子本身！为了完成这种“内部相亲”，句子里的每一个词，都要通过三个不同的“滤镜”（权重矩阵 $W^Q, W^K, W^V$），分裂出三个不同的身份：

1. 查询（Query, $\mathbf{q}$）： “我在找什么？”
   * 比如 "it" 的 $\mathbf{q}$ 向量可能在表达：“我是一个代词，我需要找一个名词，而且最好是个能感觉到‘累’的活物。”
2. 键（Key, $\mathbf{k}$）： “我是谁？我有什么特征？”
   * 比如 "animal" 的 $\mathbf{k}$ 向量会贴上标签：“我是个名词，我是个活物。”
   * "street" 的 $\mathbf{k}$ 向量会贴上标签：“我是个名词，我是个无生命物体。”
3. 值（Value, $\mathbf{v}$）： “我实际包含的信息精华是什么？”
   * 这是词语本身的深层含义，也就是最终要被提取走的内容。

#### 位置编码

对于纯粹的自注意力机制来说，“狗咬人”和“人咬狗”这三个字输入进去后，算出来的注意力分数和最终结果是一模一样的。
因为在模型眼里，这就只是三个漂浮在空间里的词汇，它们互相之间做点积打分，根本没有“谁先谁后”的概念。这就好比你把一首优美的诗剪碎了，装在一个袋子里摇匀，然后倒给计算机看。

我们需要一种完美的编码方式，必须同时满足三个苛刻的条件：

* 不管句子多长，数值都必须被死死限制在一个安全的范围里（比如 -1 到 1 之间）。
* 不管句子长短，相邻两个词之间的“距离感”必须永远保持一致。
* 模型不仅要能读懂绝对位置（我是第几个词），还能轻松算出相对位置（我离他有几个词）。

#### 正弦余弦魔法

假设我们的词向量是 512 维的，位置编码也是一个 512 维的向量。这个向量里的数字，是由不同频率的 $\sin$ 和 $\cos$ 函数生成的。

* 向量前面的几个维度（就像秒针）：波动的频率极高，相邻两个词在这里的数值差异巨大。
* 向量中间的维度（就像分针）：波动频率中等。
* 向量后面的几个维度（就像时针）：波动的频率极低，可能跨越几十个词，这里的数值才会有明显的改变。

通过这一排以不同频率转动的“时钟组合”，每个位置（Position）都拥有了一个全宇宙独一无二的、由连续数字组成的“条形码”。

对于位置为 $pos$ 的词，它的位置编码向量在第 $2i$ 和 $2i+1$ 维度的计算公式为：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

* $pos$：就是这个词在句子里的绝对位置（第 1 个词，第 2 个词...）。
* $i$：是向量的维度索引（也就是第几个“指针”）。
* $d$：是词向量的总维度（比如 512）。
* 分母那个 $10000^{2i/d}$：就是用来控制频率的。$i$ 越小（靠前的维度），分母越小，频率越高（秒针）；$i$ 越大（靠后的维度），分母极大，频率极低（时针）。

由于用的是 $\sin$ 和 $\cos$，算出来的数字永远在 $[-1, 1]$ 之间，完美解决了“数值爆炸”的问题！此外，数学上可以通过三角函数的和差化积公式证明：位置 $pos+k$ 的编码，可以表示为位置 $pos$ 的编码的线性组合。 这意味着，模型非常容易就能学会识别词与词之间的**“相对距离”**！

#### 位置拼接

有了这个 512 维的位置编码向量（身份证），我们怎么把它交给原本 512 维的词向量（比如“狗”这个词本身的含义）呢？直觉上，我们应该把它们拼起来，变成一个 1024 维的向量，一半存“意思”，一半存“位置”。但 Transformer 的做法极其狂野：直接把它们相加！

$$\text{最终输入向量} = \text{词向量 Embedding} + \text{位置编码 PE}$$

你可能会大惊失色：“把意思和位置直接加在一起？这不是把原本的语义信息污染了吗？狗都不纯粹了！”这是因为，在极高维度（如 512 维）的连续空间里，空间大到超乎想象。词义信息和位置信息在这个高维空间里，大概率是互相“垂直（正交）”的。直接相加，就相当于你在地图的二维坐标 $(X, Y)$ 上加了一个高度坐标变成 $(X, Y, Z)$，虽然数值混在一起了，但神经网络极其聪明，它完全能在后续的计算中，轻松地把“语义”和“位置”这两层信息重新剥离并分别利用起来。而且直接相加，还帮模型省掉了一半的内存！

### Transformer

先看网络结构：

![transformer](./src/transformer.svg)

区别于seq2seq,编码器中使用自注意替换RNN,增加残差连接和layer norm,，解码器第一个为自注意力，第二个为seq2seq的标准注意力。

Transformer 整体上依然保留了我们之前讲过的 Seq2Seq（编码器-解码器） 架构。

* 左半边是编码器（Encoder）： 负责阅读和理解原文（比如英文）。
* 右半边是解码器（Decoder）： 负责边看原文、边生成译文（比如中文）。

不同的是，它彻底抛弃了以前慢吞吞的 RNN，整栋大厦全靠“注意力机制”和“全连接层”搭建而成。

#### 编码器

编码器的任务，是把输入的句子（比如 "I love apples"）变成一个高度浓缩了上下文语境的“终极信息矩阵”。它是由 $N$ 个（通常是 6 个）完全相同的**“编码器块”**堆叠而成的。

现在，专注于一个编码器块：

1. 进门准备：词向量 + 位置编码：词汇先进门，换上 512 维的数字外衣（词向量），然后戴上我们上一节发的“GPS 身份证”（位置编码）。现在，带有顺序信息的词汇准备开始闯关了。
2. 第一关：多头自注意力 (Multi-Head Self-Attention)：这是我们讲过的核心！句子里的所有词开始“全场相亲”。
   * 每个词分裂出自己的 $Q, K, V$。
   * 通过多头机制，从语法、情感、逻辑等多个视角互相打分、融合。
   * 结果： 每个词都吸收了周围其他词的信息，从“孤立的词”变成了“懂语境的词”。
3. 神奇的粘合剂：残差连接与层归一化 (Add & Norm)：如果你看架构图，会发现有一条线直接绕过了注意力机制，把输入和输出加（Add）在了一起。这叫残差连接。
   * 为什么加？ 这是为了防止模型“学废了”。如果注意力机制在这个词上没学到什么好东西，直接加上原来的词向量，相当于留了一条“保底”的后路（原汁原味的信息不丢失）。
   * Norm（层归一化）： 加完之后，数字大小可能会参差不齐。Norm 就像个班主任，把所有数字重新拉回到均值为 0、方差为 1 的标准范围内，让模型训练极其稳定。
4. 第二关：前馈神经网络 (Feed Forward)：刚才的自注意力，是让词汇“互相交流”。现在的这个前馈网络（其实就是一个两层的多层感知机 MLP），是让每个词**“闭门思考”**。它单独对每个词汇的向量进行非线性变换，进一步提取更深层的特征。
5. 再次粘合：Add & Norm：思考完毕后，再次进行残差连接和归一化，确保信息稳定。

把这个块重复堆叠 6 次，编码器的工作就完成了。输出的矩阵，包含了原文极其深度的结构和语境信息。这些信息会被送到右边的解码器桌子上，准备迎接“开卷考试”。

#### 解码器

解码器的结构和编码器很像，也是 $N$ 个块堆叠，但它的内部多了一个特殊的机关。它的任务是根据编码器给的线索，以及自己已经翻译出来的词，预测下一个词。假设我们已经翻译出了“我 爱”，现在要预测“苹果”。

1. 进门准备：目标词输入 + 位置编码：把已经翻译出的“我 爱”变成向量并加上位置编码输入进去。
2. 第一关：掩蔽多头自注意力 (Masked Multi-Head Self-Attention)：
   注意这里的 Masked（掩蔽）！它和我们最早讲的过滤 \<PAD\> 废话的掩蔽不同。
   * 痛点： 翻译是一步一步来的。在训练时，为了速度，我们会把整句答案“我 爱 苹果”全喂给解码器。但如果让“爱”去做自注意力计算时看到了后面的“苹果”，那这就叫**“作弊（数据穿越）”**！
   * 做法： 引入一个因果掩码（Causal Mask）。把当前词后面所有的词的分数全部强行改成 $-\infty$。
   * 结果： “爱”在计算时，只能看到自己和前面的“我”，绝对看不到未来的“苹果”。这就保证了预测的严谨性。随后，照常进行 Add & Norm。
3. 第二关：编码器-解码器注意力 (Cross-Attention / 交叉注意力)
   * 查询（$Q$）： 来自解码器刚才的输出（当前翻译到了哪一步的语境）。
   * 键（$K$）和值（$V$）： 重点来了！$K$ 和 $V$ 全部来自左边编码器的最终输出（那张写满原文信息的桌子）。
   * 动作： 翻译官拿着现在的疑问（$Q$），去原文的桌子（$K, V$）上寻找最匹配的线索，然后把线索吸收到自己的脑子里。
   * 随后，依然是日常的 Add & Norm。
4. 第三关：前馈神经网络 (Feed Forward) + Add & Norm：和编码器一样，拿着融合了原文线索的信息，闭门思考，提炼特征，并进行最后的归一化。
5. 最终审判：线性层 + Softmax：解码器的最后一个块输出了一堆向量，但我们要的是具体的汉字。
   * 线性层（Linear）： 把向量映射到一个超级大的字典大小（比如 50,000 个汉字）。
   * Softmax： 把这 50,000 个分数变成概率百分比。
   * 概率最高的那一个，就是模型最终吐出来的词：“苹果”！

## NLP

### 预训练

#### 词嵌入(word2vec)

在自然语言处理 (NLP) 中，词嵌入是将人类语言映射到计算机能够理解的数学空间的核心技术。简单来说，它将词语转换成向量。这个向量称为词向量。顾名思义， 词向量是用于表示单词意义的向量， 并且还可以被认为是单词的特征向量或表示。

##### one-hot编码的缺陷

one-hot编码的原理：

假设词表中有 10,000 个词，"苹果"是第 5 个词，那么它的向量就是一个长度为 10,000 的向量，只有第 5 位是 1，其余全是 0。

* 苹果 = [0, 0, 0, 0, 1, ... 0]
* 香蕉 = [0, 0, 0, 0, 0, ... 1]

问题显而易见：

* 稀疏性 (Sparsity)： 向量维度极高且绝大多数是 0，浪费计算资源。
* 缺乏语义 (No Semantics)： 任何两个词的向量都是正交的（垂直的）。计算机无法知道 "苹果" 和 "香蕉" 是相似的水果，而 "苹果" 和 "汽车" 是不相关的。

对于缺乏语义的理解，可以参考书上的余弦相似度，本质上还是两个向量正交，点积为0。这里的余弦相似度，事实上是已知空间中两个向量，求这两个向量的余弦值（线性代数），其计算公式如下：

$$\text{Similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

* 结果接近 1：表示两个词非常相似（方向一致）。
* 结果接近 0：表示两个词不相关（正交）。
* 结果接近 -1：表示两个词含义相反。

词嵌入的出现，就是为了解决这些问题，将“语义”注入到向量中。它将每个词映射到一个固定长度的向量，这些向量能更好地表达不同词之间的相似性和类比关系。word2vec工具包含两个模型，即跳元模型（skip-gram）和连续词袋（CBOW）。对于在语义上有意义的表示，它们的训练依赖于条件概率，条件概率可以被看作使用语料库中一些词来预测另一些单词。由于是不带标签的数据，因此跳元模型和连续词袋都是自监督模型。

词嵌入通过训练，将每个词映射到一个低维、稠密的实数向量空间中（通常是 50 到 300 维）。在这个空间中，语义相似的词，在距离上会靠得很近。词嵌入有如下特点：

* 聚类 (Clustering)： 水果类的词会聚在一起，职业类的词会聚在另一处。
* 类比关系 (Analogy)： 这是词嵌入最著名的特性。向量之间存在线性关系，满足代数运算：
  $$\vec{King} - \vec{Man} + \vec{Woman} \approx \vec{Queen}$$

##### 跳元模型（skip-gram）

Skip-gram 的任务是——给定一个中心词，预测它周围的上下文词。

接下来举一个例子说明，假设句子是："The quick brown fox jumps over the lazy dog"设定窗口大小 (Window Size) 为 2（即看左边 2 个词，右边 2 个词）。

如果中心词选定为 "brown"：

* 输入 (Input): `brown`
* 目标 (Target/Output): 模型需要预测出 The, quick, fox, jumps 这四个词出现的概率最高。

在训练时，Skip-gram 会把上述句子拆解成一对对的 (Input, Output) 训练样本：

1. 中心词 brown -> 上下文 The $\Rightarrow$ 样本: (brown, The)
2. 中心词 brown -> 上下文 quick $\Rightarrow$ 样本: (brown, quick)
3. 中心词 brown -> 上下文 fox $\Rightarrow$ 样本: (brown, fox)
4. 中心词 brown -> 上下文 jumps $\Rightarrow$ 样本: (brown, jumps)

这就是它的特点：一个中心词，会产生多个训练样本。这也是为什么 Skip-gram 训练时间比 CBOW 长，但在数据量少时效果更好的原因。

神经网络架构：

Skip-gram 本质上是一个非常简单的三层神经网络（输入层、隐藏层、输出层）。

1. 输入层 (Input Layer)
   * 输入是一个 One-Hot 向量，代表当前的中心词。
   * 假设词汇表大小 $V = 10,000$，"brown" 是第 500 个词，那么输入向量就是一个长度 10,000 的向量，第 500 位是 1，其余是 0。
2. 隐藏层 (Hidden Layer) - 核心所在
   * 这里没有激活函数（如 ReLU 或 Sigmoid），只有线性映射。
   * 输入向量乘以一个权重矩阵 $W$（维度为 $V \times N$，其中 $N$ 是我们要设定的词向量维度，比如 300）。
   * 关键点： 因为输入是 One-Hot，这一步**本质上就是从矩阵 $W$ 中通过“查表”取出了第 500 行**。这一行，就是 "brown" 这个词目前的词向量。
3. 输出层 (Output Layer)
   * 隐藏层的输出（即 "brown" 的词向量）会乘以另一个权重矩阵 $W'$。
   * 然后通过 Softmax 函数，生成一个长度为 $V$ 的概率分布向量。
   * 目标： 我们希望在这个概率分布中，属于上下文词（如 "fox", "quick"）对应的维度概率最大。

训练优化：负采样 (Negative Sampling)

如果完全按照上面的架构训练，有一个巨大的计算瓶颈：每次预测，Softmax 都需要计算词表中所有 10,000 个词（甚至 100,000+）的概率并进行归一化。计算量太大，速度极慢。因此，引入负采样：它的思想由“预测每一个词的概率”转变为“做一个二分类任务”。

对于训练样本 (brown, fox)（正样本）：

1. 正例： 告诉模型，brown 和 fox 是搭档，输出要接近 1。
2. 负例： 随机从词表中抽取几个（比如 5 个）和 brown 不相关的词（如 "computer", "sky", "table"）。告诉模型，brown 和这些词不是搭档，输出要接近 0。

我们不再更新所有 10,000 个词的权重，每次只更新 1 个正样本 + 5 个负样本 的权重。计算量瞬间减少了几个数量级。

##### 连续词袋（CBOW）模型

CBOW 的核心思想是：根据上下文的词语来预测中间的目标词。想象你在做一个英语完形填空题：

"The quick brown fox ____ over the lazy dog."

虽然中间的词空缺了，但根据周围的词（上下文 context），你可以很容易推断出中间的词大概率是 "jumps"。

CBOW 的工作方式正是如此：

* 输入： 目标词周围的上下文词（例如 "The", "quick", "brown", "fox", "over", "the", ...）。
* 输出： 目标词的概率分布（例如 "jumps" 的概率最大）。

之所以叫“词袋（Bag-of-Words）”，是因为在输入层，模型并不在乎上下文词语的顺序，它将上下文视为一袋词的集合，取其平均值或和。

模型架构：CBOW 是一个浅层的神经网络（Shallow Neural Network），主要由三层组成：输入层、投影层（隐藏层）和输出层。

1. 输入层 (Input Layer)
   假设词汇表大小为 $V$，上下文窗口大小为 $C$（例如前后各看 2 个词，则共 4 个词）。
   * 输入是上下文中每个词的 One-hot 向量。
   * 每个 One-hot 向量的维度为 $V$。
2. 投影层 / 隐藏层 (Projection Layer)
   这是 CBOW 最关键的部分。
   * 模型通过一个共享的权重矩阵 $W$（维度为 $V \times N$，其中 $N$ 是我们想要的词向量维度，如 128 或 300），将输入的每个 One-hot 向量映射为稠密向量。
   * 由稀疏变稠密： 所有的上下文词向量在这里被累加（Sum）或者求平均（Average）。
   * “连续”的含义： 这一步将离散的单词映射到了连续的向量空间中。
   * 注意： 与传统神经网络不同，这层通常没有非线性激活函数（如 Sigmoid 或 ReLU），这大大加快了训练速度。
3. 输出层 (Output Layer)
   * 投影层的输出向量（维度 $N$）通过另一个权重矩阵 $W'$（维度 $N \times V$）映射回词汇表大小。
   * 最后使用 Softmax 函数，计算词汇表中每个词成为“中心词”的概率。

数学分析：

假设我们要预测句子中的第 $t$ 个词 $w_t$，上下文窗口大小为 $m$。

1. 查找嵌入： 对于上下文中的每个词 $w_{t-m}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+m}$，通过输入矩阵 $W$ 查找到对应的词向量 $v_c$。
2. 聚合上下文： 计算这些词向量的平均值（或和）来表示上下文向量 $\hat{v}$：
   $$\hat{v} = \frac{1}{2m} \sum_{-m \le j \le m, j \ne 0} v_{w_{t+j}}$$
3. 预测分数： 将 $\hat{v}$ 乘以输出矩阵 $W'$，得到每个词的分数 $u$。
4. 概率计算： 使用 Softmax 将分数转换为概率：
   $$P(w_t | \text{context}) = \frac{\exp({u_{w_t}})}{\sum_{k=1}^{V} \exp({u_k})}$$
5. 目标函数： 训练的目标是最大化实际目标词的对数似然概率。

训练优化技巧：

直接计算上面的 Softmax 非常慢，因为分母需要遍历整个词汇表（$V$ 可能有几十万甚至上百万）。为了加速训练，Word2Vec 引入了两种关键技术：

1. 层次 Softmax (Hierarchical Softmax):
   * 使用哈夫曼树（Huffman Tree）来替代扁平的输出层。
   * 高频词靠近根节点，低频词远离根节点。
   * 计算概率变成了从根节点走到叶子节点的路径概率乘积，复杂度从 $O(V)$ 降为 $O(\log V)$。
2. 负采样 (Negative Sampling):这是目前更常用的方法。
   * 不再计算所有词的概率，而是只更新**目标词（正样本）和少数几个随机选取的非目标词（负样本）**的权重。
   * 这将其转化为一系列二分类问题（是目标词 vs 不是目标词），极大地减少了计算量。

#### 近似训练

近似训练包含两个部分，层次 Softmax (Hierarchical Softmax)，负采样 (Negative Sampling)。其中，负采样技术已经在跳元模型中实现过，H-Softmax在词袋模型中实现过了。不再赘述。

#### 全局向量的词嵌入（GloVe）

Word2Vec 本质上是一个“预测模型”——它通过滑动窗口，利用上下文来预测中心词（或反之）。如果说 Word2Vec 是“局部观察者”（只看窗口内的词），那么 GloVe 就是“全局统计学家”。它试图结合两种方法的优点：全局矩阵分解（如 LSA） 和 局部上下文窗口（如 Word2Vec）。

在GloVe之前，词向量主要有如下两种：

1. 基于计数（Count-based）：例如 LSA（潜在语义分析）。
   * 做法：统计整个语料库中词与词的共现频率，形成一个巨大的矩阵，然后用 SVD（奇异值分解）降维。
   * 优点：利用了全局统计信息，训练快。
   * 缺点：捕捉词语类比（如 King - Man + Woman = Queen）的能力很差。
2. 基于预测（Prediction-based）：例如 Word2Vec。
   * 做法：在局部窗口内滑动，用神经网络预测。
   * 优点：捕捉语义类比的能力极强。
   * 缺点：没利用全局统计信息（它不知道这个词在整本书里出现了多少次，只知道窗口里出现了），且对语料利用率低。

GloVe： 我全都要。我要利用全局统计矩阵，但我训练出来的向量也要能做语义加减法！即，从一个简单的直觉出发：单词之间含义的差异，可以通过它们与探测词（Probe Words）共现概率的“比率”看出来。

举个经典的例子（来自原论文）：

假设我们要区分 Ice（冰） 和 Steam（蒸汽）。

| 探测词 $k$ | $P(k \mid \text{ice})$ | $P(k \mid \text{steam})$ | 比率 $\frac{P(k \mid \text{ice})}{P(k \mid \text{steam})}$ | 解释 |
| :--- | :--- | :--- | :--- | :--- |
| Solid (固体) | 高 | 低 | 很大 ($\gg 1$) | 能区分！Solid 与 Ice 相关，与 Steam 不相关 |
| Gas (气体) | 低 | 高 | 很小 ($\ll 1$) | 能区分！Gas 与 Ice 不相关，与 Steam 相关 |
| Water (水) | 高 | 高 | 接近 1 | 无法区分（都与水相关） |
| Fashion (时尚) | 低 | 低 | 接近 1 | 无法区分（都无关） |

这一比率（Ratio）才是携带语义信息的关键！我们的目标就是设计一个函数，使得词向量的点积能够拟合这个比率。

##### 数学原理

不用担心复杂的推导，我们直接看 GloVe 是如何建立“统计”与“向量”之间的桥梁的。

1. 构建共现矩阵 $X$
   首先，我们需要遍历整个语料库，构建一个巨大的矩阵 $X$。
   * $X_{ij}$ 表示：单词 $j$ 出现在单词 $i$ 的上下文中的次数。
   * 这是一个全局的统计量。
2. 模型假设
   我们希望词向量 $w_i, w_j$ 的关系能反映它们的共现概率。经过一系列数学推导（简化版），GloVe 的目标函数最终被设计为：
   $$w_i^T w_j + b_i + b_j = \log(X_{ij})$$
   * $w_i, w_j$：单词 $i$ 和 $j$ 的词向量。
   * $b_i, b_j$：偏置项（Bias）。
   * $\log(X_{ij})$：共现次数的对数。

   两个词向量的点积（衡量相似度），应该等于它们共现次数的对数。如果两个词经常一起出现（$X_{ij}$ 大），它们的向量点积就应该大。
3. 加权最小二乘回归（Weighted Least Squares）
   我们要找出一组 $w$ 和 $b$，使得上面的等式尽可能成立。于是，损失函数（Loss Function）定义为：
   $$J = \sum_{i,j=1}^{V} f(X_{ij}) (w_i^T w_j + b_i + b_j - \log(X_{ij}))^2$$
   这个公式非常好理解：
   * 括号里：预测值（向量点积）与真实值（统计对数）的差的平方（即均方误差 MSE）。
   * $f(X_{ij})$：这是一个权重函数。这非常关键！

   为什么需要权重函数 $f(X_{ij})$？
   * 高频词（如 "the", "and"）共现次数 $X_{ij}$ 极大，如果不加权，它们会主导 Loss，导致模型只学到了停用词。
   * 低频词（共现次数为 0 或 1）噪音很大，不应该过度拟合。

   GloVe 设计了一个精妙的权重函数：
   $$f(x) = \begin{cases} (x/x_{max})^\alpha & \text{if } x < x_{max} \\ 1 & \text{otherwise} \end{cases}$$
   （通常 $\alpha = 0.75$, $x_{max} = 100$）
   这个函数的作用是：
   * 如果共现次数很小，权重小（不让噪音干扰）。
   * 如果共现次数很大，权重被截断为 1（不让高频词统治世界）。

##### 训练流程

GloVe 的训练过程和 Word2Vec 略有不同：

1. 扫描全库：遍历整个语料库，统计 $X$ 矩阵（这一步最耗时，但只需做一次）。
2. 初始化：随机初始化词向量矩阵 $W$。
3. 迭代优化：遍历 $X$ 中的非零元素，使用随机梯度下降（SGD）或 AdaGrad 来最小化上面的 Loss 函数 $J$。
4. 最终向量：
   由于共现矩阵是对称的（$X_{ij} \approx X_{ji}$），理论上 $w_i$ 和 $w_j$ 是对称的。但在训练中由于随机初始化，它们会有细微差别。
   GloVe 的做法是训练两个矩阵 $W$ 和 $\tilde{W}$，最后输出：
   $$W_{final} = W + \tilde{W}$$

#### 子词嵌入

在自然语言处理（NLP）中，子词嵌入（Subword Embeddings） 是一种介于“词级别（Word-level）”和“字符级别（Character-level）”之间的文本表示和切分方法。它的核心思想是将低频词或复杂词拆分成更小的、有意义的片段（即子词），同时保留高频词的完整形式。

早期的模型通常使用词级别嵌入（如 Word2Vec），但这面临几个难以克服的挑战：

* 未登录词问题（OOV）： 词汇表大小是固定的，当模型在真实世界遇到没见过的词（如新造网络用语、拼写错误或罕见的人名）时，通常只能将其映射为无意义的 <UNK> 标签。
* 形态学信息缺失： 像 "play", "playing", "played" 包含相同的词根，但在词级别模型中，它们被当作三个完全独立、毫无关联的向量来学习。
* 显存与内存瓶颈： 为了尽可能覆盖足够多的词汇，词汇表通常需要设置得非常巨大，导致嵌入矩阵（Embedding Matrix）极其庞大，难以训练和部署。

子词嵌入通过一套有限的“积木（子词）”集合组合出无限的单词，完美平衡了上述问题。

##### FastText

FastText 是由 Facebook AI Research（FAIR）在 2016 年开源的自然语言处理工具包。它主要包含两个核心功能：生成带有子词信息的词向量（Word Embeddings） 和 高效的文本分类（Text Classification）。

1. 核心思想：字符级别的 $n$-gram (Character $n$-grams)
   FastText 最重要的创新在于：它不直接为整个单词分配一个固定的向量，而是将单词拆分为多个字符片段（$n$-gram），并为这些片段学习向量表示。
   具体拆分过程：
   为了区分作为独立单词存在的词和作为子词存在的词缀，FastText 会在单词的开头和结尾加上特殊的边界符号 < 和 >。
   假设我们要处理单词 "where"，并且设置 $n=3$（即提取长度为 3 的片段）：
   1. 添加边界符： \<where\>
   2. 提取 3-gram： \<wh, whe, her, ere, re\>
   3. 保留完整单词： 除了上述片段，模型还会将带有边界符的完整单词 \<where\> 作为一个单独的整体保留。
2. 词向量的计算方式
   在 FastText 中，一个单词的最终词向量并不是独立存在的，而是由它所包含的所有 $n$-gram 子词向量相加得到的。
   如果用数学公式来表达，假设单词 $w$ 的 $n$-gram 集合为 $G_w$，每个 $n$-gram $g$ 对应的向量表示为 $z_g$，那么单词 $w$ 的最终向量 $v_w$ 为：
   $$v_w = \sum_{g \in G_w} z_g$$
   这种设计的优势：
   1. 完美解决未登录词（OOV）问题： 如果模型在测试集中遇到了一个训练集中完全没有的新词（例如 "transformingly"），虽然完整单词没见过，但模型认识 "trans", "form", "ing", "ly" 这些 $n$-gram 碎片。FastText 可以直接调取这些碎片的向量拼凑出新词的向量。
   2. 理解形态学（Morphology）： 对于像德语、俄语、土耳其语这样拥有丰富前缀、后缀、词根变化的语言，FastText 能够极大地提高表示的准确度，因为它通过子词自然地学到了词根和词缀的语义。
3. FastText 的文本分类架构
   除了生成词向量，FastText 还是一个极其强悍的文本分类器。它的网络架构非常简单，类似于 Word2Vec 中的 CBOW 模型，但专门为分类任务做了优化。
   1. 输入层： 输入一段文本中的各个词（以及 $n$-gram）的索引。
   2. 隐藏层： 将输入文本中所有词向量进行平均化（Average Pooling），得到一个代表整篇文档的单一向量。
   3. 输出层： 将文档向量输入给一个线性分类器，输出预测的标签。
   当分类标签非常多（例如几十万个类别）时，传统的 Softmax 计算会非常缓慢。FastText 使用了一棵基于哈夫曼树（Huffman Tree）的分层 Softmax，将寻找正确分类的时间复杂度从 $O(N)$ 降到了 $O(\log N)$。这使得 FastText 能在几秒钟内训练完数百万级别的文本数据，且准确率能与复杂的深度学习模型（如 CNN/RNN）媲美。

##### BPE

相较于 FastText 强制切分固定长度的 $n$-gram，BPE 是一种完全基于数据驱动的、动态的“自底向上”的聚类算法。它最初在 1994 年被发明出来用于数据压缩，后来在 2015 年被引入 NLP 领域，用于解决机器翻译中的未登录词（OOV）问题。

1. 核心算法流程
   BPE 的训练过程就像是在玩消消乐的逆向游戏：不断找出最常相邻出现的两个基础组件，把它们“粘合”成一个更大的新组件，直到达到你规定的组件总数（词表大小）。
   具体步骤如下：
   1. 准备语料与初始化

        首先，统计训练语料中所有单词的出现频率。然后，将所有单词拆分成单个字符，并在每个单词的末尾加上一个特殊的结束符（通常用 \</w\> 表示）。为什么需要 </w>？ 因为我们需要区分位于词尾的字符和位于词中的字符。例如，"est" 作为后缀（如 highest</w>）和作为单词的一部分（如 estimate）在语义上是不同的，加上 </w> 可以让模型精准学习到词尾后缀。

   2. 统计相邻字符对（Pairs）的频率
    
        在所有被拆分的字符中，统计所有相邻两个符号同时出现的次数。

   3. 合并最高频的字符对
        
        找出频率最高的那一对符号，将它们合并成一个新的、更长的符号，并加入到我们的“词汇表”中。

   4. 循环迭代
        
        用新合并的符号替换语料中的旧符号。然后不断重复第 2 步和第 3 步，直到词汇表的大小达到了我们预先设定的目标（比如 GPT-3 的词表大小是 50,257），或者没有任何相邻对的频率大于 1。
2. 演示
   我们用一个极简的微型语料库来手动推演一遍 BPE。假设我们的语料库统计出了 4 个词及其词频：
   * `low`：5 次
   * `lowest`：2 次
   * `newer`：6 次
   * `wider`：3 次

   初始状态（全部拆为单字符）：
   * `l o w </w> `: 5
   * `l o w e s t </w> `: 2
   * `n e w e r </w> `: 6
   * `w i d e r </w> `: 3
   * 当前词表：`l, o, w, e, s, t, n, i, d, r, </w>`
   
   第 1 轮合并：
   我们扫描相邻字符，发现 e 和 r 相邻出现的次数最多（在 newer 中出现 6 次，在 wider 中出现 3 次，总计 9 次）。
   * 动作： 合并 e 和 r 为 er。记录合并规则 (e, r) -> er。
   * 语料更新： n e w er \</w\> (6次), w i d er \</w\> (3次)。

   第 2 轮合并：
   现在频率最高的是 er 和 \</w\>，共计 9 次。
   * 动作： 合并 er 和 \</w\> 为 er\</w\>。记录规则 (er, \</w\>) -> er\</w\>。
   * 语料更新： n e w er\</w\> (6次), w i d er\</w\> (3次)。 

   第 3 轮合并：
   扫描发现 n 和 e 相邻出现了 6 次。
   * 动作： 合并 n 和 e 为 ne。记录规则 (n, e) -> ne。
   * 语料更新： ne w er\</w\> (6次)。

   就这样持续下去。随着合并次数的增加，原本破碎的单字符会慢慢拼凑成常见的词缀（如 est</w>，ing</w>），最终高频词（如 low</w>）会被完整地重新拼出来，而低频词可能依然保持被切分为几个子词的状态。

虽然标准的 BPE 已经很强大了，但当面对极其庞大且包含无数罕见字符（比如各种 Emoji 表情符号、生僻的中日韩汉字）的互联网语料时，基础的 BPE 依然会导致初始词表急剧膨胀。这就引入了现代大模型使用的BBPE模型。

##### BBPE

Byte-Level BPE (BBPE):保留了 BPE 算法极其优秀的统计合并逻辑，但将其作用的基础单元从“人类理解的字符”降维到了“计算机理解的字节（Byte）”。

一、 传统 BPE 面临的“灾难”

在标准的 BPE 中，我们初始的“原子词表”是语料库中出现过的所有单个字符（Characters）。对于纯英文任务，这没有任何问题，因为英文字母、数字和标点符号加起来也就 100 多个。但如果我们想训练一个掌握全球知识的大模型，语料库中不仅有英文，还有中文、日文、阿拉伯文，甚至包含了大量的代码、特殊的数学符号以及满天飞的 Emoji 🚀, 这就导致了如下的问题：

1. 词表爆炸： 全世界的 Unicode 字符超过 14 万个。如果用传统 BPE，我们还没开始进行任何合并操作，光是把这 14 万个基础字符存下来，就已经占用了巨大的模型词表空间（而 GPT-2 的总词表大小才设定为 50,257）。
2. 未登录词（OOV）的死局： 为了节省空间，工程师通常会强行砍掉那些罕见的字符（比如某个生僻的汉字或罕见的 Emoji）。一旦砍掉，当模型在现实中遇到这些字符时，就只能无奈地输出一个 <UNK>（未知符号）。模型变成了“睁眼瞎”。

二、 BBPE 的降维打击：万物皆为 256

OpenAI 给出的绝妙解法是：放弃人类视角的 Unicode 字符，直接深入到计算机底层的 UTF-8 编码。在计算机世界里，无论是英文字母 'A'，还是汉字 '中'，亦或是 Emoji '🔥'，在底层都是一串 0 和 1。而数据的基本存储单位是字节（Byte）。一个字节由 8 个比特（Bit）组成。因此，一个字节所能表示的所有可能性只有 $2^8 = 256$ 种（从十六进制的 00 到 FF）。

在 UTF-8 编码下：

* 英文字母 `a` 占用 1 个字节：61
* 中文汉字 `字` 占用 3 个字节：E5 AD 97
* Emoji 火焰 `🔥` 占用 4 个字节：F0 9F 94 A5

BBPE 的核心逻辑就诞生了：我们不再以 14 万个 Unicode 字符作为初始状态，而是以这 256 个基础字节作为我们不可再分的绝对起点！

三、 BBPE 的运作流程

BBPE 的合并过程与标准 BPE 没有任何区别，只是起点变了。

1. 初始状态： 词表大小严格锁定为 256（加上几个特殊 token，如 \<|endoftext|\>）。无论语料库多大，包含多少种奇怪的语言，初始原子永远只有这 256 个。
2. 转译语料： 将所有的训练文本全部无情地转换为 UTF-8 字节流。
3. 统计与合并（与常规 BPE 相同）： 在字节流中，统计相邻字节对的频率，将最高频的字节对合并为更长的新字节序列。

四、举个例子

假设我们的语料中频繁出现中文词语“自然语言”。

* 最初，它们只是 12 个散落的字节：`[E8] [87] [AA] [E7] [84] [B6] [E8] [AF] [AD] [E8] [A8] [80]`
* 经过几轮 BBPE 合并，“自”的三个字节 `[E8] [87] [AA]` 因为总是连在一起出现，被合并成了一个 token `[E887AA]`。
* 合并继续进行，由于“自然”经常连用，`[E887AA]` 和 `[E784B6]` 最终被合并成了一个超大 token：`[E887AAE784B6]`（代表“自然”）。

##### WordPiece

见代码。