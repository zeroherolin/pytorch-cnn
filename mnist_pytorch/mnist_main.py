import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import struct
import os

MNIST_DIR = "../mnist_data"
TRAIN_DATA = "train-images-idx3-ubyte"
TRAIN_LABEL = "train-labels-idx1-ubyte"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"


class MNIST_MLP(object):
    def __init__(self, batch_size=100, input_size=784, hidden1=32, hidden2=16, out_classes=10, lr=0.01, max_epoch=1,
                 print_iter=100):
        # 硬件配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 参数配置
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter
        self.lowest_loss = float("inf")

        # 数据相关
        self.train_data = None
        self.test_data = None

        # 模型组件
        self.model = self.build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def load_mnist(self, file_dir, is_images=True):
        bin_file = open(file_dir, 'rb')
        bin_data = bin_file.read()
        bin_file.close()

        if is_images:
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
        else:
            fmt_header = '>ii'
            magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
            num_rows, num_cols = 1, 1
        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
        mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
        print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(mat_data.shape)))

        return mat_data

    def load_data(self):
        print('Loading MNIST data from files...')
        train_images = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_DATA), True)
        train_labels = self.load_mnist(os.path.join(MNIST_DIR, TRAIN_LABEL), False)
        test_images = self.load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
        test_labels = self.load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)

        # 确保标签数据转换为整数
        self.train_data = np.append(
            train_images.astype(np.float32),
            train_labels.astype(np.int64).reshape(-1, 1),
            axis=1
        )
        self.test_data = np.append(
            test_images.astype(np.float32),
            test_labels.astype(np.int64).reshape(-1, 1),
            axis=1
        )

    def shuffle_data(self):
        print('Randomly shuffle MNIST data...')
        np.random.shuffle(self.train_data)

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden1),
            nn.ReLU(),
            nn.Linear(self.hidden1, self.hidden2),
            nn.ReLU(),
            nn.Linear(self.hidden2, self.out_classes)
        )
        return model

    def load_model(self, param_dir):
        print('Saving parameters to file ' + param_dir)
        checkpoint = torch.load(param_dir, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.lowest_loss = checkpoint['lowest_loss']
        self.model.to(self.device)

    def save_model(self, param_dir):
        print('Loading parameters from file ' + param_dir)
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'lowest_loss': self.lowest_loss
        }, param_dir)

    def _to_tensor(self, data, dtype=torch.float32):
        tensor = torch.from_numpy(data).to(dtype)
        if dtype == torch.float32:
            tensor = tensor / 255.0  # 归一化
        return tensor.to(self.device)

    def train(self):
        max_batch = self.train_data.shape[0] // self.batch_size

        print('Start training...')
        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()
            for idx_batch in range(max_batch):
                batch = self.train_data[idx_batch * self.batch_size:(idx_batch + 1) * self.batch_size]
                batch_images = self._to_tensor(batch[:, :-1])
                batch_labels = torch.from_numpy(
                    batch[:, -1].astype(np.int64)
                ).to(self.device)
                outputs = self.model(batch_images)
                loss = self.criterion(outputs, batch_labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if idx_batch % self.print_iter == 0:
                    print(f'Epoch [{idx_epoch + 1}/{self.max_epoch}], '
                          f'Iter [{idx_batch}/{max_batch}], '
                          f'Loss: {loss.item():.6f}')
                    if loss.item() < self.lowest_loss:
                        self.lowest_loss = loss.item()
                        self.save_model('mlp-%d-%d-%depoch.pth' % (self.hidden1, self.hidden2, self.max_epoch))
                        print('Find lowest loss, model saved.')

    def evaluate(self):
        total_samples = 0
        correct = 0
        with torch.no_grad():
            for idx in range(int(self.test_data.shape[0] / self.batch_size)):
                batch = self.test_data[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_images = self._to_tensor(batch[:, :-1])

                batch_labels = torch.from_numpy(
                    batch[:, -1].astype(np.int64)
                ).to(self.device)

                outputs = self.model(batch_images)
                _, predicted = torch.max(outputs.data, 1)

                total_samples += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        accuracy = correct / total_samples
        print('Accuracy in test set:%f' % accuracy)


def build_mnist_mlp():
    h1, h2, e = 128, 64, 20
    mlp = MNIST_MLP(hidden1=h1, hidden2=h2, max_epoch=e)
    mlp.load_data()
    mlp.build_model()
    mlp.train()
    # mlp.save_model('mlp-%d-%d-%depoch.npy' % (h1, h2, e))
    mlp.load_model('mlp-%d-%d-%depoch.pth' % (h1, h2, e))
    return mlp


if __name__ == '__main__':
    mlp = build_mnist_mlp()
    mlp.evaluate()
