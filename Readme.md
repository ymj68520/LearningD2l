# 跟着李沐学AI

## 安装

首先需要安装conda～

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

| 维度        | 欠拟合     | 过拟合     |
| --------- | ------- | ------- |
| 训练误差      | 高       | 低       |
| 验证 / 测试误差 | 高       | 高       |
| 泛化间隙      | 小（但整体差） | 大       |
| 学习曲线      | 早早停滞    | 后期分叉    |
| 偏差–方差     | 高偏差、低方差 | 低偏差、高方差 |

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


