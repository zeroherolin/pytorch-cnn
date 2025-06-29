# mnist_manul/layers.py

import numpy as np


class FullyConnectedLayer(object):
    """全连接层"""

    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        print(f'\tFully connected layer: input={self.num_input}, output={self.num_output}.')

    def init_param(self, std=0.01):
        """初始化权重和偏置"""
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])

    def forward(self, input):
        """前向传播"""
        self.input = input  # 保存输入用于反向传播
        self.output = self.input.dot(self.weight) + self.bias
        return self.output

    def backward(self, top_diff):
        """反向传播"""
        # 计算权重和偏置的梯度
        self.d_weight = np.matmul(self.input.T, top_diff)
        self.d_bias = np.matmul(np.ones([1, top_diff.shape[0]]), top_diff)
        # 计算传递到前一层的梯度
        bottom_diff = np.matmul(top_diff, self.weight.T)
        return bottom_diff

    def update_param(self, lr):
        """参数更新"""
        self.weight -= lr * self.d_weight
        self.bias -= lr * self.d_bias

    def load_param(self, weight, bias):
        """加载预训练参数"""
        self.weight = weight
        self.bias = bias

    def save_param(self):
        """保存参数"""
        return self.weight, self.bias


class ReLULayer(object):
    """ReLU激活层"""

    def __init__(self):
        print('\tReLU layer.')

    def forward(self, input):
        """前向传播：ReLU(x) = max(0, x)"""
        self.input = input  # 保存输入用于反向传播
        return np.maximum(0, self.input)

    def backward(self, top_diff):
        """反向传播：梯度只通过正数区域"""
        return top_diff * (self.input >= 0.)


class SoftmaxLossLayer(object):
    """Softmax损失层"""

    def __init__(self):
        print('\tSoftmax loss layer.')

    def forward(self, input):
        """前向传播：计算softmax概率"""
        # 数值稳定处理：减去最大值防止指数爆炸
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob

    def get_loss(self, label):
        """计算交叉熵损失"""
        self.batch_size = self.prob.shape[0]
        # 创建one-hot编码标签
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0

        # 计算平均交叉熵损失
        loss = -np.sum(np.log(self.prob + 1e-8) * self.label_onehot) / self.batch_size
        return loss

    def backward(self):
        """反向传播：计算softmax层的梯度"""
        return (self.prob - self.label_onehot) / self.batch_size
