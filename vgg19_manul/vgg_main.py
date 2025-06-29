# vgg19_manul/vgg_main.py

import numpy as np
import scipy.io
from PIL import Image

from layers import ConvolutionalLayer, ReLULayer, MaxPoolingLayer, FlattenLayer, FullyConnectedLayer, SoftmaxLossLayer


class VGG19(object):
    """VGG19网络实现"""

    def __init__(self, param_path='../assets/imagenet-vgg-verydeep-19.mat'):
        self.param_path = param_path
        # 定义网络层名称顺序
        self.param_layer_name = (
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
            'flatten', 'fc6', 'relu6', 'fc7', 'relu7', 'fc8', 'softmax'
        )

    def build_model(self):
        """构建VGG19网络结构"""
        print('Building VGG-19 model...')
        self.layers = {}

        # Block 1
        self.layers['conv1_1'] = ConvolutionalLayer(3, 3, 64, 1, 1)
        self.layers['relu1_1'] = ReLULayer()
        self.layers['conv1_2'] = ConvolutionalLayer(3, 64, 64, 1, 1)
        self.layers['relu1_2'] = ReLULayer()
        self.layers['pool1'] = MaxPoolingLayer(2, 2)

        # Block 2
        self.layers['conv2_1'] = ConvolutionalLayer(3, 64, 128, 1, 1)
        self.layers['relu2_1'] = ReLULayer()
        self.layers['conv2_2'] = ConvolutionalLayer(3, 128, 128, 1, 1)
        self.layers['relu2_2'] = ReLULayer()
        self.layers['pool2'] = MaxPoolingLayer(2, 2)

        # Block 3
        self.layers['conv3_1'] = ConvolutionalLayer(3, 128, 256, 1, 1)
        self.layers['relu3_1'] = ReLULayer()
        self.layers['conv3_2'] = ConvolutionalLayer(3, 256, 256, 1, 1)
        self.layers['relu3_2'] = ReLULayer()
        self.layers['conv3_3'] = ConvolutionalLayer(3, 256, 256, 1, 1)
        self.layers['relu3_3'] = ReLULayer()
        self.layers['conv3_4'] = ConvolutionalLayer(3, 256, 256, 1, 1)
        self.layers['relu3_4'] = ReLULayer()
        self.layers['pool3'] = MaxPoolingLayer(2, 2)

        # Block 4
        self.layers['conv4_1'] = ConvolutionalLayer(3, 256, 512, 1, 1)
        self.layers['relu4_1'] = ReLULayer()
        self.layers['conv4_2'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu4_2'] = ReLULayer()
        self.layers['conv4_3'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu4_3'] = ReLULayer()
        self.layers['conv4_4'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu4_4'] = ReLULayer()
        self.layers['pool4'] = MaxPoolingLayer(2, 2)

        # Block 5
        self.layers['conv5_1'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu5_1'] = ReLULayer()
        self.layers['conv5_2'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu5_2'] = ReLULayer()
        self.layers['conv5_3'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu5_3'] = ReLULayer()
        self.layers['conv5_4'] = ConvolutionalLayer(3, 512, 512, 1, 1)
        self.layers['relu5_4'] = ReLULayer()
        self.layers['pool5'] = MaxPoolingLayer(2, 2)

        # 分类头
        self.layers['flatten'] = FlattenLayer()
        self.layers['fc6'] = FullyConnectedLayer(25088, 4096)
        self.layers['relu6'] = ReLULayer()
        self.layers['fc7'] = FullyConnectedLayer(4096, 4096)
        self.layers['relu7'] = ReLULayer()
        self.layers['fc8'] = FullyConnectedLayer(4096, 1000)
        self.layers['softmax'] = SoftmaxLossLayer()

        # 需要参数更新的层列表
        self.update_layer_list = []
        for layer_name in self.layers.keys():
            if 'conv' in layer_name or 'fc' in layer_name:
                self.update_layer_list.append(layer_name)

    def init_model(self):
        """初始化模型参数"""
        print('Initializing VGG-19 parameters...')
        for layer_name in self.update_layer_list:
            self.layers[layer_name].init_param()

    def load_model(self):
        """加载预训练模型参数"""
        print(f'Loading parameters from {self.param_path}')
        params = scipy.io.loadmat(self.param_path)

        # 获取并打印图像均值
        self.image_mean = params['normalization'][0][0][0]
        self.image_mean = np.mean(self.image_mean, axis=(0, 1))
        print(f'Image mean: {self.image_mean}')

        # 按顺序加载各层参数
        for idx in range(43):
            layer_name = self.param_layer_name[idx]

            # 处理卷积层参数
            if 'conv' in layer_name:
                weight, bias = params['layers'][0][idx][0][0][0][0]
                weight = np.transpose(weight, [2, 0, 1, 3])  # 调整维度顺序
                bias = bias.reshape(-1)
                self.layers[layer_name].load_param(weight, bias)

            # 处理全连接层参数
            if idx >= 37 and 'fc' in layer_name:
                weight, bias = params['layers'][0][idx - 1][0][0][0][0]
                weight = weight.reshape([weight.shape[0] * weight.shape[1] * weight.shape[2], weight.shape[3]])
                self.layers[layer_name].load_param(weight, bias)

    def load_image(self, image_dir):
        """加载并预处理输入图像"""
        print(f'Loading and preprocessing image from {image_dir}')
        img = Image.open(image_dir)
        img = img.resize((224, 224))  # VGG输入尺寸为224x224
        self.input_image = np.array(img).astype(np.float32)

        # 减去图像均值
        self.input_image -= self.image_mean

        # 调整维度顺序 [H, W, C] -> [C, H, W]
        self.input_image = np.transpose(self.input_image, [2, 0, 1])

        # 添加批次维度 [1, C, H, W]
        self.input_image = np.expand_dims(self.input_image, axis=0)

    def forward(self):
        """执行前向传播"""
        print('Starting forward propagation...')
        current = self.input_image

        # 按顺序通过各层
        for layer_name in self.param_layer_name:
            print(f'Processing layer: {layer_name}')
            current = self.layers[layer_name].forward(current)

        return current

    def evaluate(self):
        """评估模型：执行推理并输出结果"""
        prob = self.forward()
        top1 = np.argmax(prob[0])
        print(f'Classification result: id={top1}, probability={prob[0, top1]:.4f}')


if __name__ == '__main__':
    vgg = VGG19()
    vgg.build_model()
    vgg.init_model()
    vgg.load_model()
    vgg.load_image('../assets/cat1.jpeg')
    vgg.evaluate()
