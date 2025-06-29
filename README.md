# 基于Pytorch实现神经网络

## 快速跳转

- [Mnist训练和推理](#Mnist训练和推理)
- [VGG19模型推理](#VGG19模型推理)

## 模型下载

```bash
cd assets
wget http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
```

***

## Mnist训练和推理

网络结构：
```mermaid
graph LR
    输入层((输入图像<br>28x28像素)) --> 全连接层1{全连接层<br>输出维度: 128}
    全连接层1 --> ReLU激活函数层1[ReLU激活函数层]
    ReLU激活函数层1 --> 全连接层2{全连接层<br>输出维度: 64}
    全连接层2 --> ReLU激活函数层2[ReLU激活函数层]
    ReLU激活函数层2 --> 全连接层3{全连接层<br>输出维度: 10}
    全连接层3 --> Softmax损失层[Softmax损失层]
    Softmax损失层 --> 交叉熵损失[交叉熵损失]
```

训练流：
```mermaid
graph LR
    subgraph 数据加载模块
        输入数据 --> 数据读取和预处理
        数据读取和预处理 --> 样本队列
    end
    subgraph 网络结构模块
        神经网络初始化 --> 建立网络结构
        建立网络结构 --> 神经网络参数初始化
    end
    subgraph 网络训练模块
        样本队列 --> 前向传播
        神经网络参数初始化 --> 模型参数
        前向传播 --> 反向传播
        反向传播 --> 参数更新
        参数更新 --> 模型参数
        模型参数 --> 前向传播
        模型参数 --> 保存神经网络参数
    end
```

推理流：
```mermaid
graph LR
    subgraph 数据加载模块
        输入数据 --> 数据读取和预处理
        数据读取和预处理 --> 样本队列
    end
    subgraph 网络结构模块
        神经网络初始化 --> 建立网络结构
    end
    subgraph 网络推理模块
        样本队列 --> 前向传播
        前向传播 --> 计算精度
        建立网络结构 --> 加载神经网络参数
        加载神经网络参数 --> 模型参数
        模型参数 --> 前向传播
    end
```

### 基于Numpy手动实现

- [mnist_manul](mnist_manul/)

### 基于Pytorch框架实现

- [mnist_pytorch](mnist_pytorch/)

## VGG19模型推理

网络结构：
```mermaid
graph LR
    L1 --> L2
    L2 --> L3
    L3 --> L4
    L4 --> L5
    L5 --> L6
    subgraph L1
        卷积层1_1 --> ReLU激活函数层1_1
        ReLU激活函数层1_1 --> 卷积层1_2
        卷积层1_2 --> ReLU激活函数层1_2
        ReLU激活函数层1_2 --> 最大池化层1
    end
    subgraph L2
%%        最大池化层1 --> 卷积层2_1
        卷积层2_1 --> ReLU激活函数层2_1
        ReLU激活函数层2_1 --> 卷积层2_2
        卷积层2_2 --> ReLU激活函数层2_2
        ReLU激活函数层2_2 --> 最大池化层2
    end
    subgraph L3
%%        最大池化层2 --> 卷积层3_1
        卷积层3_1 --> ReLU激活函数层3_1
        ReLU激活函数层3_1 --> 卷积层3_2
        卷积层3_2 --> ReLU激活函数层3_2
        ReLU激活函数层3_2 --> 卷积层3_3
        卷积层3_3 --> ReLU激活函数层3_3
        ReLU激活函数层3_3 --> 卷积层3_4
        卷积层3_4 --> ReLU激活函数层3_4
        ReLU激活函数层3_4 --> 最大池化层3
    end
    subgraph L4
%%        最大池化层3 --> 卷积层4_1
        卷积层4_1 --> ReLU激活函数层4_1
        ReLU激活函数层4_1 --> 卷积层4_2
        卷积层4_2 --> ReLU激活函数层4_2
        ReLU激活函数层4_2 --> 卷积层4_3
        卷积层4_3 --> ReLU激活函数层4_3
        ReLU激活函数层4_3 --> 卷积层4_4
        卷积层4_4 --> ReLU激活函数层4_4
        ReLU激活函数层4_4 --> 最大池化层4
    end
    subgraph L5
%%        最大池化层4 --> 卷积层5_1
        卷积层5_1 --> ReLU激活函数层5_1
        ReLU激活函数层5_1 --> 卷积层5_2
        卷积层5_2 --> ReLU激活函数层5_2
        ReLU激活函数层5_2 --> 卷积层5_3
        卷积层5_3 --> ReLU激活函数层5_3
        ReLU激活函数层5_3 --> 卷积层5_4
        卷积层5_4 --> ReLU激活函数层5_4
        ReLU激活函数层5_4 --> 最大池化层5
    end
    subgraph L6
%%        最大池化层5 --> 全连接层6
        全连接层6 --> ReLU激活函数层6
        ReLU激活函数层6 --> 全连接层7
        全连接层7 --> ReLU激活函数层7
        ReLU激活函数层7 --> 全连接层8
        全连接层8 --> Softmax损失层
    end
```

### 基于Numpy手动实现

- [vgg19_manul](vgg19_manul/)

### 基于Pytorch框架实现

- [vgg19_pytorch](vgg19_pytorch/)

## Updating...
