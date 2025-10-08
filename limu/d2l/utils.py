
#训练一轮模型的函数
import torch
from torch import nn
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import numpy as np
import struct
from IPython.display import clear_output  # 用于动态绘图

import IPython.display as Idisplay
# -------------------------- 1. 手动实现：本地加载Fashion-MNIST（替代d2l.load_mnist_local） --------------------------
def load_mnist_local(batch_size, root="/home/pumengyu/2025_9python/download/"):
    """
    本地解析Fashion-MNIST二进制文件，返回训练集和测试集的DataLoader（替代d2l.load_mnist_local）
    root: 数据集根路径（需包含 FashionMNIST/raw 子目录）
    batch_size: 批量大小
    """
    # 定义数据集解析函数（内部子函数，处理单集（训练/测试））
    def parse_mnist(subset="train"):
        # 1. 拼接数据集文件路径
        data_dir = os.path.join(root, "FashionMNIST", "raw")
        if subset == "train":
            img_path = os.path.join(data_dir, "train-images-idx3-ubyte")
            label_path = os.path.join(data_dir, "train-labels-idx1-ubyte")
        else:
            img_path = os.path.join(data_dir, "t10k-images-idx3-ubyte")
            label_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte")

        # 2. 解析图像文件（MNIST格式：前16字节为头信息）
        with open(img_path, "rb") as f:
            # 读取头信息：magic数、图像数、行数、列数（大端字节序）
            magic, num_imgs, rows, cols = struct.unpack(">IIII", f.read(16))
            # 读取像素数据→转NumPy→转Tensor→调整形状+归一化（[0,1]）
            img_bytes = f.read()
            imgs = np.frombuffer(img_bytes, dtype=np.uint8)
            imgs = torch.from_numpy(imgs.copy()).view(num_imgs, 1, rows, cols).float() / 255.0

        # 3. 解析标签文件（MNIST格式：前8字节为头信息）
        with open(label_path, "rb") as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            label_bytes = f.read()
            labels = torch.from_numpy(np.frombuffer(label_bytes, dtype=np.uint8).copy()).long()

        # 4. 封装为Dataset
        class FashionMNISTDataset(data.Dataset):
            def __len__(self):
                return len(imgs)
            def __getitem__(self, idx):
                return imgs[idx], labels[idx]

        return FashionMNISTDataset()

    # 加载训练集和测试集
    train_dataset = parse_mnist(subset="train")
    test_dataset = parse_mnist(subset="test")

    # 5. 封装为DataLoader（与原d2l返回格式一致：train_iter, test_iter）
    train_iter = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_iter = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4  # 测试集无需shuffle
    )

    return train_iter, test_iter


# -------------------------- 2. 工具类1：累加器（统计训练指标，如正确数、总样本数） --------------------------
class Accumulator:
    """在n个变量上累加（替代d2l.Accumulator）"""
    def __init__(self, n):
        self.data = [0.0] * n  # 初始化n个累加变量（如[正确数, 总样本数]）

    def add(self, *args):
        self.data = [a + float(b.detach() if isinstance(b,torch.Tensor) else b) for a, b in zip(self.data, args)]  # 累加更新

    def reset(self):
        self.data = [0.0] * len(self.data)  # 重置

    def __getitem__(self, idx):
        return self.data[idx]  # 按索引获取累加值


# -------------------------- 3. 工具类2：动态绘图器（替代d2l.Animator） --------------------------
def set_axes(axes, xlabel=None, ylabel=None, xlim=None, ylim=None, legend=None):
    """配置坐标轴（辅助Animator，替代d2l.set_axes）"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid(True, linestyle='--', alpha=0.7)  # 显示网格


class Animator:
    """动态绘制训练曲线（损失、准确率），无d2l依赖"""
    def __init__(self, xlabel='epoch', ylabel='value', legend=None, xlim=[1, 10], ylim=[0.3, 0.9]):
        # 初始化画布
        plt.rcParams['savefig.format'] = 'svg'  # 图像格式（清晰）
        self.fig, self.axes = plt.subplots(figsize=(5, 3.5))
        # 配置坐标轴
        self.config_axes = lambda: set_axes(self.axes, xlabel, ylabel, xlim, ylim, legend)
        # 存储数据（多条曲线的x/y坐标）
        self.X, self.Y, self.fmts = None, None, ('-', 'm--', 'g-.')  # 线条格式

    def add(self, x, y):
        """添加数据点并刷新图像"""
        # 统一y的格式（确保为列表，支持多条曲线）
        if not hasattr(y, '__len__'):
            y = [y]
        n = len(y)
        # 统一x的格式（与y长度匹配）
        if not hasattr(x, '__len__'):
            x = [x] * n

        # 初始化数据存储列表
        if self.X is None:
            self.X = [[], ] * n
        if self.Y is None:
            self.Y = [[], ] * n

        # 追加新数据
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

        # 清空旧图→绘制新图→配置坐标轴
        self.axes.cla()
        for x_line, y_line, fmt in zip(self.X, self.Y, self.fmts):
            self.axes.plot(x_line, y_line, fmt)
        self.config_axes()

        # 动态刷新显示
        Idisplay.display(self.fig)
        Idisplay.clear_output(wait=True)


# -------------------------- 4. 辅助函数1：计算模型准确率（替代d2l.evaluate_accuracy） --------------------------
def evaluate_accuracy(net, data_iter):
    """计算模型在指定数据集上的准确率"""
    if isinstance(net, nn.Module):
        net.eval()  # 模型设为评估模式（禁用Dropout等）
    metric = Accumulator(2)  # 统计：[正确预测数, 总样本数]
    with torch.no_grad():  # 禁用梯度计算（节省内存）
        for X, y in data_iter:
            # 预测：取概率最大的类别（argmax(axis=1)）
            y_hat = net(X)
            if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
                y_hat = y_hat.argmax(axis=1)
            # 统计正确数（预测类别与真实类别一致）
            cmp = y_hat.type(y.dtype) == y
            metric.add(cmp.type(y.dtype).sum(), y.numel())
    return metric[0] / metric[1]  # 准确率 = 正确数 / 总样本数

def train_epoch_ch3_regression(net, train_iter, loss, optimizer):
    """回归任务的单轮训练：仅计算MSE损失并更新参数"""
    if isinstance(net, nn.Module):
        net.train()  # 切换为“训练模式”（若有Dropout/BatchNorm等需此步骤）
    metric = Accumulator(2)  # 存储：[总损失, 总样本数]
    for X, y in train_iter:
        y_hat = net(X)       # 前向传播：预测值
        l = loss(y_hat, y)   # 计算每个样本的MSE损失
        optimizer.zero_grad()  # 清空历史梯度
        l.mean().backward()    # 损失均值反向传播（避免梯度爆炸）
        optimizer.step()       # 更新模型参数
        metric.add(l.sum(), y.numel())  # 累加“总损失”和“样本数”
    return metric[0] / metric[1]  # 返回“平均损失”
# -------------------------- 5. 辅助函数2：训练一轮模型（替代d2l.train_epoch_ch3） --------------------------
def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一轮，返回平均损失和训练准确率"""
    if isinstance(net, nn.Module):
        net.train()  # 模型设为训练模式（启用Dropout等）
    metric = Accumulator(3)  # 统计：[总损失, 正确预测数, 总样本数]
    for X, y in train_iter:
        # 1. 前向传播：计算预测值和损失
        y_hat = net(X)
        l = loss(y_hat, y)  # loss为nn.CrossEntropyLoss(reduction='none')，输出每个样本的损失

        # 2. 反向传播+参数更新
        if isinstance(updater, torch.optim.Optimizer):
            # 若updater是PyTorch优化器（如SGD）
            updater.zero_grad()  # 清空梯度
            l.mean().backward()  # 损失均值反向传播（避免梯度爆炸）
            updater.step()       # 更新参数
        else:
            # 若updater是自定义优化器（此处用不到，保留兼容性）
            l.sum().backward()
            updater(X.shape[0])

        # 3. 累加指标
        metric.add(l.sum(),  # 总损失
                   (y_hat.argmax(axis=1).type(y.dtype) == y).type(y.dtype).sum(),  # 正确数
                   y.numel())  # 总样本数

    # 返回：平均损失（总损失/总样本数）、训练准确率（正确数/总样本数）
    return metric[0] / metric[2], metric[1] / metric[2]


# -------------------------- 6. 核心函数1：完整训练逻辑（替代d2l.train_ch3） --------------------------
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型并动态显示进度（无d2l依赖）"""
    # 初始化动态绘图器（显示：训练损失、训练准确率、测试准确率）
    animator = Animator(
        xlabel='epoch', ylabel='value',
        legend=['train loss', 'train acc', 'test acc'],
        xlim=[1, num_epochs], ylim=[0.2, 1.0]  # 调整y轴范围，适配实际值
    )

    for epoch in range(num_epochs):
        # 1. 训练一轮，获取指标
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss, updater)
        # 2. 计算测试集准确率
        test_acc = evaluate_accuracy(net, test_iter)
        # 3. 动态添加数据点
        animator.add(epoch + 1, (train_loss, train_acc, test_acc))

        # （可选）断言验证训练有效性（放宽阈值，避免早期训练不满足报错）
        assert train_loss < 10.0, f"训练损失过高：{train_loss}"
        assert train_acc > 0.5, f"训练准确率过低：{train_acc}"
        assert test_acc > 0.5, f"测试准确率过低：{test_acc}"

    # 最终输出指标
    train_loss, train_acc = train_epoch_ch3(net, train_iter, loss, updater)
    test_acc = evaluate_accuracy(net, test_iter)
    print(f"\n最终指标：")
    print(f"训练损失：{train_loss:.4f} | 训练准确率：{train_acc:.4f} | 测试准确率：{test_acc:.4f}")


# -------------------------- 7. 辅助函数3：标签转换与图像显示（替代d2l相关函数） --------------------------
def get_fashion_mnist_labels(labels):
    """将Fashion-MNIST数字标签转为文本标签（如0→'t-shirt'）"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """显示多张图像（替代d2l.show_images）"""
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()  # 展平为1D列表，方便遍历
    for ax, img, title in zip(axes, imgs, titles):
        # 显示灰度图像（挤压通道维度：(1,28,28)→(28,28)）
        ax.imshow(img.numpy().squeeze(), cmap="gray")
        ax.axis("off")  # 隐藏坐标轴
        if title:
            ax.set_title(title, fontsize=8)  # 设置标签（小字体避免重叠）
    return fig


# -------------------------- 8. 核心函数2：预测并显示结果（替代d2l.predict_ch3） --------------------------
def predict_ch3(net, test_iter, n=6):
    """预测测试集图像类别，并显示“真实标签vs预测标签”（无d2l依赖）"""
    # 从测试集中取一个批次（仅取前n个样本）
    for X, y in test_iter:
        break  # 只取第一个批次

    # 1. 模型预测
    net.eval()  # 评估模式
    with torch.no_grad():
        y_hat = net(X)
        y_hat_labels = y_hat.argmax(axis=1)  # 预测类别

    # 2. 转换标签为文本
    true_labels = get_fashion_mnist_labels(y[:n])  # 真实标签（前n个）
    pred_labels = get_fashion_mnist_labels(y_hat_labels[:n])  # 预测标签（前n个）

    # 3. 组合标签（真实标签\n预测标签，换行显示）
    titles = [f"True: {t}\nPred: {p}" for t, p in zip(true_labels, pred_labels)]

    # 4. 显示图像
    fig = show_images(
        imgs=X[:n], num_rows=1, num_cols=n,
        titles=titles, scale=1.2
    )
    plt.show()  # 显式触发显示
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_array(data_arrays, batch_size, is_train=True):
    """
    将numpy数组转换为PyTorch的DataLoader迭代器
    
    参数:
        data_arrays: 包含特征和标签的元组 (X, y)，均为numpy数组
        batch_size: 批量大小
        is_train: 是否为训练集（训练集需要打乱数据，测试集不需要）
    
    返回:
        DataLoader: 可迭代的数据加载器
    """
    # 将numpy数组转换为PyTorch张量，并设置数据类型为float32
    tensors = tuple(torch.tensor(data, dtype=torch.float32) for data in data_arrays)
    
    # 创建数据集
    dataset = TensorDataset(*tensors)
    
    # 创建数据加载器，训练集打乱数据，测试集不打乱
    return DataLoader(dataset, batch_size, shuffle=is_train)
#暂退法

def dropout(x, dropout_prob):
    # x为输入张量，dropout_prob为丢弃概率
    if dropout_prob == 0:
        return x
    
    # 生成与x形状相同的掩码，值为0或1
    a=np.random.normal(x.shape)
    
    mask = a < (1 - dropout_prob)
    
    # 应用掩码并缩放输出（除以1 - dropout_prob以保持期望不变）
    return (x * mask) / (1 - dropout_prob)
a=torch.arange(10)
print(dropout(a,0.1))
# -------------------------- 9. 原有模型代码（无需修改，直接调用上述自定义函数） --------------------------
if __name__ == "__main__":
    # 1. 加载数据（调用自定义load_mnist_local，无d2l依赖）
    batch_size = 256
    train_iter, test_iter = load_mnist_local(batch_size)

    # 2. 初始化模型参数（原有代码不变）
    num_inputs, num_outputs, num_hiddens = 784, 10, 256  # 784=28*28（图像展平）
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)  # 隐藏层权重
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))  # 隐藏层偏置
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)  # 输出层权重
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))  # 输出层偏置
    params = [W1, b1, W2, b2]

    # 3. 激活函数（原有代码不变）
    def relu(X):
        return torch.max(X, torch.zeros_like(X))  # ReLU：max(X, 0)

    # 4. 模型定义（原有代码不变）
    #def net(X):
        #X = X.reshape(-1, num_inputs)  # 图像展平：(batch,1,28,28)→(batch,784)
        #H = relu(torch.matmul(X, W1) + b1)  # 隐藏层：X@W1 + b1 → ReLU
        #return torch.matmul(H, W2) + b2  # 输出层：H@W2 + b2（无softmax，CrossEntropyLoss自带）
    # 2. 初始化模型（替换原“初始化模型参数”+“激活函数”+“模型定义”）
    class MLP(nn.Module):  # 继承nn.Module，成为PyTorch标准模型类
        def __init__(self, num_inputs, num_hiddens, num_outputs):
            super(MLP, self).__init__()  # 调用父类构造函数
            # 定义模型参数（用nn.Parameter包裹，自动纳入模型参数管理）
            self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
            self.b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
            self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
            self.b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
        
        def forward(self, X):  # 前向传播逻辑（原net函数的核心逻辑移到这里）
            X = X.reshape(-1, num_inputs)  # 图像展平：(batch,1,28,28)→(batch,784)
            H = torch.relu(torch.matmul(X, self.W1) + self.b1)  # 用torch自带relu，更高效
            return torch.matmul(H, self.W2) + self.b2  # 输出层

        # 3. 实例化模型（替换原“def net(X)”）
    num_inputs, num_hiddens, num_outputs = 784, 256, 10  # 784=28*28
    net = MLP(num_inputs, num_hiddens, num_outputs)  # 现在net是nn.Module实例，有eval()方法
        # 5. 损失函数与优化器（原有代码不变）
    loss = nn.CrossEntropyLoss(reduction='none')  # 交叉熵损失（每个样本单独计算）
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(net.parameters(), lr=lr)  # PyTorch内置SGD优化器

    # 6. 训练模型（调用自定义train_ch3）
    print("开始训练...")
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

    # 7. 预测并显示结果（调用自定义predict_ch3）
    print("\n预测结果展示：")
    predict_ch3(net, test_iter, n=6)  # 显示6个测试样本