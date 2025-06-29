# mnist_pytorch/mnist_main.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import struct
import os

# 常量定义
MNIST_DIR = "../assets/mnist_data"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"


class MNIST_MLP:
    """使用PyTorch实现的MNIST分类多层感知机"""

    def __init__(self, batch_size=100, input_size=784,
                 hidden1=128, hidden2=64, out_classes=10,
                 lr=0.01, max_epoch=20, print_iter=100):
        # 硬件配置：自动检测GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 网络参数配置
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter
        self.lowest_loss = float("inf")  # 跟踪最低损失

        # 数据存储
        self.train_data = None
        self.test_data = None

        # 初始化模型组件
        self.model = self.build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()  # 包含softmax的交叉熵损失
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

    def load_mnist(self, file_dir, is_images=True):
        """加载MNIST二进制数据文件"""
        with open(file_dir, 'rb') as bin_file:
            bin_data = bin_file.read()

        # 解析文件头
        if is_images:
            # 图像文件头格式：魔数|图像数|行数|列数
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        else:
            # 标签文件头格式：魔数|标签数
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
            num_rows = num_cols = 1  # 标签无空间维度

        # 解析数据主体
        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
        mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])

        print(f'Loaded {num_images} samples from {file_dir}, shape: {mat_data.shape}')
        return mat_data

    def load_data(self):
        """加载并预处理MNIST数据集"""
        print('Loading MNIST data...')
        train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
        train_labels = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABEL), False)
        test_images = self.load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
        test_labels = self.load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)

        # 合并特征和标签，确保数据类型正确
        self.train_data = np.column_stack((
            train_images.astype(np.float32),
            train_labels.astype(np.int64)
        ))
        self.test_data = np.column_stack((
            test_images.astype(np.float32),
            test_labels.astype(np.int64)
        ))
        print(f'Train data shape: {self.train_data.shape}, Test data shape: {self.test_data.shape}')

    def shuffle_data(self):
        """打乱训练数据顺序"""
        print('Shuffling training data...')
        np.random.shuffle(self.train_data)

    def build_model(self):
        """构建神经网络架构"""
        print(f'Building MLP model: {self.input_size}-{self.hidden1}-{self.hidden2}-{self.out_classes}')
        model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden1),
            nn.ReLU(),
            nn.Linear(self.hidden1, self.hidden2),
            nn.ReLU(),
            nn.Linear(self.hidden2, self.out_classes)
        )
        return model

    def load_model(self, param_dir):
        """加载预训练模型"""
        if not os.path.exists(param_dir):
            print(f"Model {param_dir} not found. Training from scratch.")
            return False

        print(f'Loading model from {param_dir}')
        checkpoint = torch.load(param_dir, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.lowest_loss = checkpoint['lowest_loss']
        self.model.to(self.device)
        return True

    def save_model(self, param_dir):
        """保存模型参数和优化器状态"""
        print(f'Saving model to {param_dir}')
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'lowest_loss': self.lowest_loss
        }, param_dir)

    def _to_tensor(self, data, is_image=True):
        """将numpy数组转换为PyTorch张量并进行预处理"""
        tensor = torch.from_numpy(data)
        if is_image:
            tensor = tensor.float() / 255.0  # 归一化到[0,1]
            tensor = tensor.to(self.device)
        else:
            tensor = tensor.long().to(self.device)
        return tensor

    def train(self):
        """训练模型的主循环"""
        max_batch = self.train_data.shape[0] // self.batch_size
        print(f'Starting training for {self.max_epoch} epochs...')

        # 训练循环
        for epoch in range(self.max_epoch):
            self.shuffle_data()  # 每轮打乱数据
            running_loss = 0.0

            for batch_idx in range(max_batch):
                # 准备批次数据
                start = batch_idx * self.batch_size
                end = (batch_idx + 1) * self.batch_size
                batch = self.train_data[start:end]

                # 分离图像和标签
                images = self._to_tensor(batch[:, :-1])
                labels = self._to_tensor(batch[:, -1], is_image=False)

                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 统计损失
                running_loss += loss.item()

                # 定期打印训练状态
                if batch_idx % self.print_iter == 0:
                    avg_loss = running_loss / self.print_iter
                    print(f'Epoch [{epoch + 1}/{self.max_epoch}], '
                          f'Batch [{batch_idx}/{max_batch}], '
                          f'Loss: {avg_loss:.6f}')

                    # 检查并保存最佳模型
                    if avg_loss < self.lowest_loss:
                        self.lowest_loss = avg_loss
                        self.save_model('mnist-best.pth')
                        print(f'New lowest loss ({avg_loss:.6f}), model saved.')

                    running_loss = 0.0

    def evaluate(self):
        """在测试集上评估模型性能"""
        print('Evaluating on test set...')
        self.model.eval()  # 设置为评估模式
        total_samples = 0
        correct = 0

        with torch.no_grad():  # 禁用梯度计算
            # 分批处理测试集
            for idx in range(0, self.test_data.shape[0], self.batch_size):
                # 准备批次数据
                batch = self.test_data[idx:idx + self.batch_size]
                images = self._to_tensor(batch[:, :-1])
                labels = self._to_tensor(batch[:, -1], is_image=False)

                # 前向传播
                outputs = self.model(images)

                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 计算并打印最终准确率
        accuracy = 100 * correct / total_samples
        print(f'Test Accuracy: {accuracy:.2f}% ({correct}/{total_samples})')
        self.model.train()  # 恢复训练模式


def build_mnist_mlp():
    """构建并训练MNIST MLP模型"""
    mlp = MNIST_MLP(
        hidden1=128,
        hidden2=64,
        max_epoch=20,
        lr=0.01,
        print_iter=20
    )
    mlp.load_data()

    # 尝试加载已有模型，否则训练
    if not mlp.load_model('mnist-best.pth'):
        mlp.train()

    return mlp


if __name__ == '__main__':
    # 构建、训练和评估模型
    mlp = build_mnist_mlp()
    mlp.evaluate()
