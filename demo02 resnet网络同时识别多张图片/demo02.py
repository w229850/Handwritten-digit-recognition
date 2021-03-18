import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy
from torch.nn import functional as F
import glob
import os

class ResBlk(nn.Module):  # 定义Resnet Block模块
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):  # 进入网络前先得知道传入层数和传出层数的设定
        """
        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()  # 初始化

        # we add stride support for resbok, which is distinct from tutorials.
        # 根据resnet网络结构构建2个（block）块结构 第一层卷积 卷积核大小3*3,步长为1，边缘加1
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        # 将第一层卷积处理的信息通过BatchNorm2d
        self.bn1 = nn.BatchNorm2d(ch_out)
        # 第二块卷积接收第一块的输出，操作一样
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        # 确保输入维度等于输出维度
        self.extra = nn.Sequential()  # 先建一个空的extra
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):  # 定义局部向前传播函数
        """

        :param x: [b, ch, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))  # 对第一块卷积后的数据再经过relu操作
        out = self.bn2(self.conv2(out))  # 第二块卷积后的数据输出
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out  # 将x传入extra经过2块（block）输出后与原始值进行相加
        out = F.relu(out)  # 调用relu，这里使用F.调用

        return out


class ResNet18(nn.Module):  # 构建resnet18层

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(  # 首先定义一个卷积层
            nn.Conv2d(1, 32, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(32)
        )
        # followed 4 blocks 调用4次resnet网络结构，输出都是输入的2倍
        # [b, 64, h, w] => [b, 128, h ,w]
        self.blk1 = ResBlk(32, 64, stride=1)
        # [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = ResBlk(64, 128, stride=1)
        # # [b, 256, h, w] => [b, 512, h, w]
        self.blk3 = ResBlk(128, 256, stride=1)
        # # [b, 512, h, w] => [b, 1024, h, w]
        self.blk4 = ResBlk(256, 256, stride=1)

        self.outlayer = nn.Linear(256 * 1 * 1, 10)  # 最后是全连接层

    def forward(self, x):  # 定义整个向前传播
        """
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))  # 先经过第一层卷积

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)  # 然后通过4次resnet网络结构
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # print('after conv:', x.shape) #[b, 512, 2, 2]
        # F.adaptive_avg_pool2d功能尾巴变为1,1，[b, 512, h, w] => [b, 512, 1, 1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print('after pool:', x.shape)
        x = x.view(x.size(0), -1)  # 平铺一维值
        x = self.outlayer(x)  # 全连接层

        return x

# 加载网络和参数
model = ResNet18()
model.load_state_dict(torch.load('model.ckpt'))
model.eval()

file_list = glob.glob(os.path.join('./number1/','*'))
grid_rows = len(file_list) / 5 +1

for i, file in enumerate(file_list):
    image = Image.open(file).resize((28,28))
    gray_image = image.convert("L")
    transform = transforms.ToTensor()
    im_data = transform(gray_image)
    im_data = im_data.resize(1,1,28,28)
    outputs = model(im_data)
    _, pred = torch.max(outputs, 1)
    plt.subplot(grid_rows, 5, i + 1)
    plt.imshow(gray_image)
    plt.title("is {}".format(pred.item()), fontsize=24)
    plt.axis('off')
    print('[{}]预测数字为: [{}]'.format(file, pred.item()))
plt.show()

