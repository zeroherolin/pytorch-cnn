# vgg19_manul/layers.py

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


class ConvolutionalLayer(object):
    """卷积层"""

    def __init__(self, kernel_size, channel_in, channel_out, padding, stride):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        print(f'\tConvolutional layer: kernel={kernel_size}, '
              f'in_channel={channel_in}, out_channel={channel_out}.')

    def init_param(self, std=0.01):
        """初始化参数"""
        self.weight = np.random.normal(
            loc=0.0,
            scale=std,
            size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out)
        )
        self.bias = np.zeros([self.channel_out])

    def forward(self, input):
        """前向传播"""
        self.input = input
        # 计算填充后尺寸
        height = self.input.shape[2] + 2 * self.padding
        width = self.input.shape[3] + 2 * self.padding

        # 填充输入
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :,
        self.padding:self.padding + self.input.shape[2],
        self.padding:self.padding + self.input.shape[3]] = self.input

        # 计算输出尺寸
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (height - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])

        # 执行卷积操作
        for idxn in range(self.input.shape[0]):  # 批次维度
            for idxc in range(self.channel_out):  # 输出通道维度
                for idxh in range(height_out):  # 高度维度
                    for idxw in range(width_out):  # 宽度维度
                        # 提取当前感受野
                        receptive_field = self.input_pad[idxn, :,
                                          idxh * self.stride:idxh * self.stride + self.kernel_size,
                                          idxw * self.stride:idxw * self.stride + self.kernel_size]

                        # 计算卷积结果
                        self.output[idxn, idxc, idxh, idxw] = np.sum(
                            self.weight[:, :, :, idxc] * receptive_field
                        ) + self.bias[idxc]
        return self.output

    def load_param(self, weight, bias):
        """加载预训练参数"""
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias


class MaxPoolingLayer(object):
    """最大池化层"""

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        print(f'\tMax pooling layer: kernel={kernel_size}, stride={stride}.')

    def forward(self, input):
        """前向传播"""
        self.input = input  # [N, C, H, W]
        # 计算输出尺寸
        height_out = (self.input.shape[2] - self.kernel_size) // self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])

        # 执行池化操作
        for idxn in range(self.input.shape[0]):  # 批次维度
            for idxc in range(self.input.shape[1]):  # 通道维度
                for idxh in range(height_out):  # 高度维度
                    for idxw in range(width_out):  # 宽度维度
                        # 提取当前池化区域
                        pool_region = self.input[idxn, idxc,
                                      idxh * self.stride:idxh * self.stride + self.kernel_size,
                                      idxw * self.stride:idxw * self.stride + self.kernel_size]

                        # 计算最大值
                        self.output[idxn, idxc, idxh, idxw] = np.max(pool_region)
        return self.output


class FlattenLayer(object):
    """展平层"""

    def __init__(self):
        print('\tFlatten layer.')

    def forward(self, input):
        """前向传播：将多维特征图展平为一维向量"""
        # 调整维度顺序 [N, H, W, C]
        self.input = np.transpose(input, [0, 2, 3, 1])
        # 计算输出形状 [N, H*W*C]
        self.output_shape = [np.prod(self.input.shape[1:])]
        # 重塑为二维矩阵 [N, H*W*C]
        self.output = self.input.reshape([self.input.shape[0]] + self.output_shape)
        return self.output
