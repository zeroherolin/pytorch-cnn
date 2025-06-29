# mnist_manul/mnist_main.py

import numpy as np
import struct
import os
from layers import FullyConnectedLayer, ReLULayer, SoftmaxLossLayer

# 常量定义
MNIST_DIR = "../assets/mnist_data"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"


class MNIST_MLP:
    """MNIST分类的多层感知机"""

    def __init__(self, batch_size=100, input_size=784,
                 hidden1=128, hidden2=64, out_classes=10,
                 lr=0.01, max_epoch=20, print_iter=100):
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
        """加载全部MNIST数据集"""
        print('Loading MNIST data...')
        train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
        train_labels = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABEL), False)
        test_images = self.load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
        test_labels = self.load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)

        # 合并特征和标签
        self.train_data = np.c_[train_images, train_labels]
        self.test_data = np.c_[test_images, test_labels]

    def shuffle_data(self):
        """打乱训练数据顺序"""
        print('Shuffling data...')
        np.random.shuffle(self.train_data)

    def build_model(self):
        """构建网络结构"""
        print('Building MLP model:')
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(self.hidden1, self.hidden2)
        self.relu2 = ReLULayer()
        self.fc3 = FullyConnectedLayer(self.hidden2, self.out_classes)
        self.softmax = SoftmaxLossLayer()
        self.update_layer_list = [self.fc1, self.fc2, self.fc3]  # 需要更新的层

    def init_model(self):
        """初始化模型参数"""
        print('Initializing parameters...')
        for layer in self.update_layer_list:
            layer.init_param()

    def load_model(self, param_dir):
        """加载预训练模型"""
        if not os.path.exists(param_dir):
            print(f"Model {param_dir} not found. Training from scratch.")
            return False

        print(f'Loading model from {param_dir}')
        params = np.load(param_dir, allow_pickle=True).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])
        self.fc3.load_param(params['w3'], params['b3'])
        return True

    def save_model(self, param_dir):
        """保存模型参数"""
        print(f'Saving model to {param_dir}')
        params = {
            'w1': self.fc1.weight, 'b1': self.fc1.bias,
            'w2': self.fc2.weight, 'b2': self.fc2.bias,
            'w3': self.fc3.weight, 'b3': self.fc3.bias
        }
        np.save(param_dir, params)

    def forward(self, input):
        """前向传播"""
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        h2 = self.relu2.forward(h2)
        h3 = self.fc3.forward(h2)
        return self.softmax.forward(h3)

    def backward(self):
        """反向传播（从最后一层开始）"""
        dloss = self.softmax.backward()
        dh2 = self.fc3.backward(dloss)
        dh1 = self.relu2.backward(dh2)
        dh1 = self.fc2.backward(dh1)
        dh0 = self.relu1.backward(dh1)
        _ = self.fc1.backward(dh0)

    def update(self, lr):
        """更新所有层参数"""
        for layer in self.update_layer_list:
            layer.update_param(lr)

    def train(self):
        """训练模型"""
        max_batch = self.train_data.shape[0] // self.batch_size
        print(f'Start training for {self.max_epoch} epochs...')

        for epoch in range(self.max_epoch):
            self.shuffle_data()  # 每轮打乱数据

            for batch_idx in range(max_batch):
                # 获取当前批次数据
                start = batch_idx * self.batch_size
                end = (batch_idx + 1) * self.batch_size
                batch_images = self.train_data[start:end, :-1]
                batch_labels = self.train_data[start:end, -1].astype(int)

                # 前向传播 + 计算损失
                _ = self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)

                # 反向传播 + 参数更新
                self.backward()
                self.update(self.lr)

                # 定期打印日志
                if batch_idx % self.print_iter == 0:
                    print(f'Epoch [{epoch + 1}/{self.max_epoch}], '
                          f'Batch [{batch_idx}/{max_batch}], '
                          f'Loss: {loss:.6f}')

                    # 保存最佳模型
                    if loss < self.lowest_loss:
                        self.lowest_loss = loss
                        self.save_model('mnist-best.npy')
                        print(f'New lowest loss ({loss:.6f}), model saved.')

    def evaluate(self):
        """在测试集上评估模型"""
        print('Evaluating on test set...')
        num_samples = self.test_data.shape[0]
        pred_results = np.zeros(num_samples)

        # 分批处理测试集
        for idx in range(0, num_samples, self.batch_size):
            batch_images = self.test_data[idx:idx + self.batch_size, :-1]
            prob = self.forward(batch_images)
            pred_labels = np.argmax(prob, axis=1)
            pred_results[idx:idx + self.batch_size] = pred_labels

        # 计算准确率
        accuracy = np.mean(pred_results == self.test_data[:, -1])
        print(f'Test Accuracy: {accuracy * 100:.2f}%')


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
    mlp.build_model()

    # 尝试加载已有模型，否则训练
    if not mlp.load_model('mnist-best.npy'):
        mlp.init_model()
        mlp.train()

    return mlp


if __name__ == '__main__':
    mlp = build_mnist_mlp()
    mlp.evaluate()
