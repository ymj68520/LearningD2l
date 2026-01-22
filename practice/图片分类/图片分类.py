#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图片分类 - 叶子图像分类项目
从 Jupyter Notebook 转换而来
"""

import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import os
from PIL import Image
import torchvision.models as models
from tqdm.auto import tqdm 
from torch.utils.data import random_split
from d2l import torch as d2l
from IPython.display import display, clear_output

# ==================== 配置参数 ====================
base_dir = "classify-leaves/"
out_dir = "classify-leaves/out/"

# ==================== 数据加载和探索 ====================
train_data = pd.read_csv(os.path.join(base_dir, "train.csv"))
val_data = pd.read_csv(os.path.join(base_dir, "test.csv"))

# 统计每个类别的数量
label_counts = train_data['label'].value_counts()

# print(train_data.shape, '\n')
# print(train_data.info(), '\n')
# print(train_data.describe(), "\n")
# print(train_data.head(), '\n')

# ==================== 数据可视化（可选） ====================
def visualize_sample_images():
    """查看前20张图片"""
    preview_image_paths = train_data['image'][:20]
    preview_image_labels = train_data['label'][:20]
    # 创建子图
    fig, axes = plt.subplots(4, 5, figsize=(12, 8))  # 一行多列

    for i, (ax, img_path) in enumerate(zip(axes.flatten(), preview_image_paths)):
        img = mpimg.imread(os.path.join(base_dir, img_path))  # 读取图片
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(preview_image_labels[i], y=-0.2)

    plt.show()

# 如需可视化，取消下面的注释
# visualize_sample_images()

# ==================== 标签映射 ====================
# 获取所有唯一的类别（叶子种类）
unique_labels = train_data["label"].unique()

# 创建 类别 → 索引 的映射
label2idx = {label: idx for idx, label in enumerate(unique_labels)}

# 创建反向映射（id → label）
idx2label = {v: k for k, v in label2idx.items()}

# ==================== 数据集定义 ====================
class LeaveDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(base_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)  # 应用转换
        
        label_name = self.data.iloc[idx, 1]
        label = label2idx[label_name]  # 转换为整数索引

        return image, label  # 返回 (图片, 标签)

# ==================== 数据预处理和加载器 ====================
batch_size = 64  # GoogLeNet使用较小的batch size以适应内存

# 定义数据转换
transform = T.Compose([
    # T.Resize((224, 224)),  # 调整图片大小
    T.ToTensor(),  # 转为张量
    # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

train_dataset = LeaveDataset(train_data, transform)

# 计算拆分大小
total_size = len(train_dataset)
test_size = int(0.2 * total_size)  # 20% 作为测试集
train_size = total_size - test_size  # 剩下的作为训练集

# 随机拆分数据
train_subset, test_subset = random_split(train_dataset, [train_size, test_size])

# 重新创建 DataLoader
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)

# print("总批次:", len(train_loader))

# ==================== 模型定义 ====================
# 加载 GoogLeNet
googlenet = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)

# 修改最终输出的类别
num_classes = len(label_counts)  
googlenet.fc = torch.nn.Linear(googlenet.fc.in_features, num_classes)

# 设备类型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

googlenet = googlenet.to(device)

# print(googlenet)

# ==================== 损失函数和优化器 ====================
# 交叉熵损失（适用于分类问题）
crossentropy = nn.CrossEntropyLoss()

# 超参数设置
lr = 0.001  # 降低学习率，因为使用了预训练权重
momentum = 0.9
weight_decay = 1e-4  # 添加L2正则化

# 优化器 - 使用Adam优化器，对GoogLeNet效果更好
optimizer = optim.Adam(googlenet.parameters(), lr=lr, weight_decay=weight_decay)
# 如果想用SGD：optimizer = optim.SGD(googlenet.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

# 学习率调度器 - 每7个epoch将学习率乘以0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ==================== 训练和验证函数 ====================
def train():
    """单次训练"""
    googlenet.train()

    batch_nums = len(train_loader)  # 批次数
    batch_size = train_loader.batch_size  # 批量大小
    size = len(train_loader.dataset)  # 数据集大小
    
    train_loss, correct = 0.0, 0.0  # 统计损失和准确率

    with tqdm(train_loader, desc="Training", unit="batch") as p:
        for X, y in p:
            X, y = X.to(device), y.to(device)
            pred = googlenet(X)
            # GoogLeNet在训练时可能返回多个输出（包括辅助分类器），只取主输出
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = crossentropy(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            p.set_postfix(loss=f"{loss.item():>8f}")  # 显示损失值
            
            train_loss += loss.item()  # 累计每个批次的平均损失
            correct += (pred.argmax(1) == y).sum().item()  # 计算正确预测的数量

    train_loss /= batch_nums
    correct /= size
    print(f"Train Accuracy: {(100*correct):>0.2f}%, Train Avg loss: {train_loss:>8f}")

    return train_loss, correct


def test():
    """验证"""
    googlenet.eval()  # 评估模式
    
    batch_nums = len(test_loader)  # 批次数
    batch_size = test_loader.batch_size  # 批量大小
    size = len(test_loader.dataset)  # 数据集大小
    
    test_loss, correct = 0.0, 0.0  # 统计损失和准确率

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = googlenet(X)
            loss = crossentropy(pred, y)

            test_loss += loss.item()  # 累计每个批次的平均损失
            correct += (pred.argmax(1) == y).sum().item()  # 计算正确预测的数量

    test_loss /= batch_nums
    correct /= size
    print(f"Test Accuracy: {(100*correct):>0.1f}%, Test Avg loss: {test_loss:>8f}")
    
    return test_loss, correct

# ==================== 主训练循环 ====================
def main_training():
    """主训练函数"""
    # 训练损失和准确率
    train_losses, train_accs = [], []

    # 测试损失和准确率
    test_losses, test_accs = [], []

    epochs = 20  # 增加训练轮数以充分训练

    best_acc = 0.0  # 记录最佳准确率
    save_path = 'best_model.pth'  # 保存路径

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        train_loss, train_acc = train()
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        test_loss, test_acc = test()
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")

        # 保存最好的模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(googlenet.state_dict(), save_path)  # 仅保存状态字典
            print(f'New best model saved with accuracy: {best_acc:.4f}')

        print("-" * 30)

    # 绘制训练过程中的损失和准确率
    plot_training_history(epochs, train_losses, test_losses, train_accs, test_accs, best_acc)
    
    return save_path, best_acc


def plot_training_history(epochs, train_losses, test_losses, train_accs, test_accs, best_acc):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 绘制损失曲线
    axes[0].plot(range(1, epochs+1), train_losses, label='Train Loss', marker='o')
    axes[0].plot(range(1, epochs+1), test_losses, label='Test Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True)

    # 绘制准确率曲线
    axes[1].plot(range(1, epochs+1), [acc*100 for acc in train_accs], label='Train Accuracy', marker='o')
    axes[1].plot(range(1, epochs+1), [acc*100 for acc in test_accs], label='Test Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    print(f'最佳测试准确率: {best_acc*100:.2f}%')

# ==================== 模型推理 ====================
def load_best_model(save_path):
    """加载最佳模型"""
    # 重新定义模型（确保架构一致）
    model = models.googlenet()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # 加载训练好的权重
    model.load_state_dict(torch.load(save_path, weights_only=True))

    # 切换到 eval 模式
    model.to(device)
    model.eval()
    print("模型已加载并设置为评估模式！")
    
    return model


def predict_single_image(model, image_path):
    """单张图片预测"""
    model.eval()

    image_val = Image.open(image_path)
    image_val_tensor = transform(image_val).unsqueeze(0)  # 应用转换 升维

    # 预测
    with torch.no_grad():
        image_val_tensor = image_val_tensor.to(device)
        output = model(image_val_tensor)
        probabilities = F.softmax(output, dim=1)  # 转换为概率
        pred_class = torch.argmax(probabilities).item()

    print(torch.max(probabilities).item() * 100, '%')
    print(idx2label[pred_class])
    
    return idx2label[pred_class], torch.max(probabilities).item()


# ==================== 批量预测 ====================
class LeaveValDataset(Dataset):
    """验证数据集（无标签）"""
    def __init__(self, data_df, transform=None):
        self.data = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(base_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)  # 应用转换
        
        return image


def batch_predict(model, output_file='submission.csv'):
    """批量预测并保存结果"""
    val_dataset = LeaveValDataset(val_data, transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print("val batch num:", len(val_loader))

    all_preds = []

    with torch.no_grad():
        for inputs in tqdm(val_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())

    print("批量预测结果长度:", len(all_preds))

    pred_labels = [idx2label[pred_id] for pred_id in all_preds]  # 转换成label
    val_data['label'] = pred_labels
    val_data.to_csv(output_file, index=False)
    print(f"预测结果已保存到 {output_file}")


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 执行训练
    print("开始训练...")
    save_path, best_acc = main_training()
    
    # 加载最佳模型
    print("\n加载最佳模型...")
    best_model = load_best_model(save_path)
    
    # 单张图片预测示例
    print("\n单张图片预测示例:")
    img_val_path = os.path.join(base_dir, val_data['image'][0])
    predict_single_image(best_model, img_val_path)
    
    # 批量预测
    print("\n开始批量预测...")
    batch_predict(best_model)
    
    print("\n所有任务完成！")
