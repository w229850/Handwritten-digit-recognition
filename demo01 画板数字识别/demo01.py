import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def test_mydata():
    # 调整图片大小
    im = plt.imread('new009.jpg')  # 读入图片
    images = Image.open('new009.jpg')    # 将图片存储到images里面
    images = images.resize((28,28))   # 调整图片的大小为28*28
    images = images.convert('L')   # 灰度化

    transform = transforms.ToTensor()
    images = transform(images)
    images = images.resize(1,1,28,28)

    # 加载网络和参数
    model = ConvNet()
    model.load_state_dict(torch.load('model.ckpt'))
    model.eval()
    outputs = model(images)

    values, indices = outputs.data.max(1) # 返回最大概率值和下标
    plt.title('{}'.format((int(indices[0]))))
    plt.imshow(im)
    plt.show()

test_mydata()
