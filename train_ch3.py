import matplotlib.pyplot as plt
from IPython import display
import torch
from d2l import torch as d2l


class Animator:
    """在训练过程中动态绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None,
                 xlim=None, ylim=None):
        self.fig, self.ax = plt.subplots()
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.xlim = xlim
        self.ylim = ylim
        self.X = []
        self.Y = [[] for _ in legend]

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if xlim:
            self.ax.set_xlim(xlim)
        if ylim:
            self.ax.set_ylim(ylim)
        if legend:
            self.ax.legend(legend)

    def add(self, x, y):
        self.X.append(x)
        for i, yi in enumerate(y):
            self.Y[i].append(yi)

        self.ax.cla()
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        if self.xlim:
            self.ax.set_xlim(self.xlim)
        if self.ylim:
            self.ax.set_ylim(self.ylim)

        for i in range(len(self.Y)):
            self.ax.plot(self.X, self.Y[i])

        if self.legend:
            self.ax.legend(self.legend)

        display.clear_output(wait=True)
        display.display(self.fig)
        plt.show()

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
        net.eval()  # 评估模式：关闭 dropout，使用固定的 batchnorm 统计
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
        net.train()  # 训练模式：开启 dropout，batchnorm 使用当前批次统计
    metric = d2l.Accumulator(3)  # 训练损失总和，训练准确率总和，样本数
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

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.0, 1.0],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    print(f'最后一轮训练损失: {train_loss:.6f}, 训练准确率: {train_acc:.6f}, 测试准确率: {test_acc:.6f}')