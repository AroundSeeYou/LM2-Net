# -------------------------------------------- #
# （2）网络构建
'''
block： 残差单元
blocks_num： 每个残差结构使用残差单元的数量
num_classes： 分类数量
include_top： 是否包含分类层（全连接）
'''
import torch
from torch import nn


# -------------------------------------------- #
class ResNet(nn.Module):
    # 初始化
    def __init__(self, block=3, blocks_num=3, num_classes=3, include_top=True):
        # 继承父类初始化方法
        super(ResNet, self).__init__()

        # 属性分配
        self.include_top = include_top
        self.in_channel = 64  # 第一个卷积层的输出通道数

        # 7*7卷积下采样层处理输入图像 [b,3,h,w]==>[b,64,h,w]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel,
                               kernel_size=7, stride=2, padding=3, bias=False)

        # BN对每个通道做标准化
        self.bn1 = nn.BatchNorm2d(self.in_channel)

        # relu激活函数
        self.relu = nn.ReLU(inplace=True)

        # 3*3最大池化层 [b,64,h,w]==>[b,64,h//2,w//2]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差卷积块
        # 第一个残差结构不需要下采样只需要调整通道
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        # 下面的残差结构的第一个残差单元需要进行下采样
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # 分类层
        if self.include_top:
            # 自适应全局平均池化，无论输入特征图的shape是多少，输出特征图的(h,w)==(1,1)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output
            # 全连接分类
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 卷积层权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    # 残差结构
    '''
    block: 代表残差单元
    channel: 残差结构中第一个卷积层的输出通道数
    block_num: 代表一个残差结构包含多少个残差单元
    stride: 是否下采样stride=2
    '''

    def _make_layer(self, block, channel, block_num, stride=1):

        # 是否需要进行下采样
        downsample = None

        # 如果stride=2或者残差单元的输入和输出通道数不一致
        # 就对残差单元的shortcut部分执行下采样操作
        if stride != 1 or self.in_channel != channel * block.expansion:
            # 残差边需要下采样
            downsample = nn.Sequential(
                # 对于第一个残差单元的残差边部分只需要调整通道
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        # 一个残差结构堆叠多个残差单元
        layers = []
        # 先堆叠第一个残差单元，因为这个需要下采样
        layers.append(block(self.in_channel, channel, stride=stride, downsample=downsample))

        # 获得第一个残差单元的输出特征图个数, 作为第二个残差单元的输入
        self.in_channel = channel * block.expansion

        # 堆叠剩下的残差单元，此时的shortcut部分不需要下采样
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        # 返回构建好了的残差结构
        return nn.Sequential(*layers)  # *代表将layers以非关键字参数的形式返还

    # 前向传播
    def forward(self, x):
        # 输入层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 残差结构
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 分类层
        if self.include_top:
            # 全局平均池化
            x = self.avgpool(x)
            # 打平
            x = torch.flatten(x, 1)
            # 全连接分类
            x = self.fc(x)

        return x