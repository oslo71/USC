
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
# from .net_sphere import *


debug = False


# 实现一个Bottleneck的类，初始化需要输入通道数与GrowthRate这两个参数
class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        # 通常1x1卷积的通道数为GrowthRate的4倍
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm3d(nChannels)
        self.conv1 = nn.Conv3d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(interChannels)
        self.conv2 = nn.Conv3d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        # 将输入x同计算的结果out进行通道拼接
        out = torch.cat((x, out), 1)
        return out


class Denseblock(nn.Module):
    def __init__(self, nChannels, growthRate, nDenseBlocks):
        super(Denseblock, self).__init__()
        layers = []
        # 将每一个Bottleneck利用nn.Sequential()整合起来，输入通道数需要线性增长
        for i in range(int(nDenseBlocks)):
            layers.append(Bottleneck(nChannels, growthRate))
            nChannels += growthRate
        self.denseblock = nn.Sequential(*layers)

    def forward(self, x):
        return self.denseblock(x)


class ResCBAMLayer(nn.Module):
    def __init__(self, in_planes, feature_size):
        super(ResCBAMLayer, self).__init__()
        self.in_planes = in_planes
        self.feature_size = feature_size
        self.ch_AvgPool = nn.AvgPool3d(feature_size, feature_size)  # kernel = stride = feature_size
        self.ch_MaxPool = nn.MaxPool3d(feature_size, feature_size)
        self.ch_Linear1 = nn.Linear(in_planes, in_planes // 4, bias=False)  # 全连接层
        self.ch_Linear2 = nn.Linear(in_planes // 4, in_planes, bias=False)
        self.ch_Softmax = nn.Softmax(1)
        self.sp_Conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)  # spatial attention 中使用 两层（Maxpool+avgpool）变为一层
        self.sp_Softmax = nn.Softmax(1)
        self.sp_sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_ch_avg_pool = self.ch_AvgPool(x).view(x.size(0), -1)
        x_ch_max_pool = self.ch_MaxPool(x).view(x.size(0), -1)
        # x_ch_avg_linear = self.ch_Linear2(self.ch_Linear1(x_ch_avg_pool))
        a = self.ch_Linear1(x_ch_avg_pool)
        x_ch_avg_linear = self.ch_Linear2(a)

        x_ch_max_linear = self.ch_Linear2(self.ch_Linear1(x_ch_max_pool))
        ch_out = (self.ch_Softmax(x_ch_avg_linear + x_ch_max_linear).view(x.size(0), self.in_planes, 1, 1, 1)) * x

        x_sp_max_pool = torch.max(ch_out, 1, keepdim=True)[0]
        x_sp_avg_pool = torch.sum(ch_out, 1, keepdim=True) / self.in_planes
        sp_conv1 = torch.cat([x_sp_max_pool, x_sp_avg_pool], dim=1)  # 按维数1（列）拼接
        sp_out = self.sp_Conv(sp_conv1)
        sp_out = self.sp_sigmoid(sp_out.view(x.size(0), -1)).view(x.size(0), 1, x.size(2), x.size(3), x.size(4))

        out = sp_out * x + x
        return out


def make_conv3d(in_channels: int, out_channels: int, kernel_size: typing.Union[int, tuple], stride: int,  # typing.Union表示此处参数为int, tuple均可
                padding: int, dilation=1, groups=1,
                bias=True) -> nn.Module:
    """
    produce a Conv3D with Batch Normalization and ReLU

    :param in_channels: num of in in
    :param out_channels: num of out channels
    :param kernel_size: size of kernel int or tuple
    :param stride: num of stride
    :param padding: num of padding
    :param bias: bias
    :param groups: groups
    :param dilation: dilation
    :return: my conv3d module
    """
    module = nn.Sequential(

        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                  groups=groups,
                  bias=bias),
        nn.BatchNorm3d(out_channels),
        nn.ReLU())
    return module


def conv3d_same_size(in_channels, out_channels, kernel_size, stride=1,
                     dilation=1, groups=1,
                     bias=True):
    padding = kernel_size // 2
    return make_conv3d(in_channels, out_channels, kernel_size, stride,
                       padding, dilation, groups,
                       bias)

# conv3d_same_size与conv3d_pooling，只输出channel不一致，conv3d_pooling不改变channel
def conv3d_pooling(in_channels, kernel_size, stride=1,
                   dilation=1, groups=1,
                   bias=False):
    padding = kernel_size // 2
    return make_conv3d(in_channels, in_channels, kernel_size, stride,
                       padding, dilation, groups,
                       bias)


class ResidualBlock(nn.Module):
    """
    a simple residual block
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.my_conv1 = make_conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.my_conv2 = make_conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = make_conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        out1 = self.conv3(inputs)
        out = self.my_conv1(inputs)
        out = self.my_conv2(out)
        out = out + out1
        return out


class ConvRes(nn.Module):
    def __init__(self, config):
        super(ConvRes, self).__init__()
        self.conv1 = conv3d_same_size(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = conv3d_same_size(in_channels=32, out_channels=32, kernel_size=3)
        self.config = config
        self.last_channel = 32
        self.first_cbam = ResCBAMLayer(32, 64)  # 以上部分对应到模型图片中的stage1&2

        # for p in self.parameters():
        #     p.requires_grad = False

        layers = []
        i = 0
        for stage in config:  # [64, 64, 64], [128, 128, 256], [256, 256, 256, 512] 这里三个stage分别对应stage3、4、5
            i = i+1
            layers.append(conv3d_pooling(self.last_channel, kernel_size=3, stride=2))
            for channel in stage:
                layers.append(ResidualBlock(self.last_channel, channel))
                self.last_channel = channel  # 重新赋值channel
            layers.append(ResCBAMLayer(self.last_channel, 64//(2**i)))
            layers.append(nn.Dropout(0.3))
            # if i<2:
            #     for p in self.parameters():
            #         p.requires_grad = False

        self.layers = nn.Sequential(*layers)
        self.avg_pooling = nn.AvgPool3d(kernel_size=3, stride=2)
        self.adapavg_pooling = nn.AdaptiveAvgPool3d(1)
        self.adapmax_pooling = nn.AdaptiveMaxPool3d(1)
        # self.fc = AngleLinear(in_features=self.last_channel, out_features=2)  # 此处原为out_features=2(单标签分类为2；多标签为num（label）)
        self.fc7 = nn.Linear(in_features=self.last_channel, out_features=6)
        self.activation = nn.Sigmoid()

    def forward(self, inputs):
        if debug:
            print(inputs.size())
        out = self.conv1(inputs)
        if debug:
            print(out.size())
        out = self.conv2(out)
        if debug:
            print(out.size())
        out = self.first_cbam(out)  # 以上stage1&2
        out = self.layers(out)  # 以上stage3、4、5
        if debug:
            print(out.size())
        # out = self.avg_pooling(out)
        # out = self.adapavg_pooling(out)
        out = self.adapmax_pooling(out)
        out = out.view(out.size(0), -1)
        if debug:
            print(out.size())
        out = self.fc7(out)
        out = self.activation(out)
        return out


class ConvDense(nn.Module):
    def __init__(self, config):
        super(ConvDense, self).__init__()
        self.denseblock_num = 4
        self.growth_rate = 16
        self.conv1 = conv3d_same_size(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = conv3d_same_size(in_channels=32, out_channels=32, kernel_size=3)
        self.config = config
        self.last_channel = 32
        self.first_cbam = ResCBAMLayer(32, 64)  # 以上部分对应到模型图片中的stage1&2

        # for p in self.parameters():
        #     p.requires_grad = False

        layers = []
        i = 0
        for stage in config:  # [64, 64, 64], [128, 128, 256], [256, 256, 256, 512] 这里三个stage分别对应stage3、4、5
            i = i + 1
            layers.append(conv3d_pooling(self.last_channel, kernel_size=3, stride=2))
            for channel in stage:
                layers.append(Denseblock(self.last_channel, self.growth_rate, self.denseblock_num))
                self.last_channel = self.denseblock_num * self.growth_rate + self.last_channel  # 重新赋值channel
            layers.append(ResCBAMLayer(self.last_channel, 64//(2**i)))

            layers.append(nn.Dropout(0.3))

            # if i<2:
            #     for p in self.parameters():
            #         p.requires_grad = False

        self.layers = nn.Sequential(*layers)
        self.avg_pooling = nn.AvgPool3d(kernel_size=3, stride=2)
        self.adapavg_pooling = nn.AdaptiveAvgPool3d(1)
        self.adapmax_pooling = nn.AdaptiveMaxPool3d(1)
        # self.fc = AngleLinear(in_features=self.last_channel, out_features=2)  # 此处原为out_features=2(单标签分类为2；多标签为num（label）)
        self.fc7 = nn.Linear(in_features=self.last_channel, out_features=6)
        self.fc1 = nn.Linear(in_features=self.last_channel, out_features=1)
        self.activation = nn.Sigmoid()

    def forward(self, inputs):
        if debug:
            print(inputs.size())
        out = self.conv1(inputs)
        if debug:
            print(out.size())
        out = self.conv2(out)
        if debug:
            print(out.size())
        out = self.first_cbam(out)  # 以上stage1&2
        out = self.layers(out)  # 以上stage3、4、5
        if debug:
            print(out.size())
        # out = self.avg_pooling(out)
        # out = self.adapavg_pooling(out)
        out = self.adapmax_pooling(out)
        out = out.view(out.size(0), -1)  #  20230128 看这一句
        if debug:
            print(out.size())
        out = self.fc7(out)
        out = self.activation(out)
        return out


def test():
    global debug
    debug = True
    net = ConvRes([[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]])
    inputs = torch.randn((1, 1, 32, 32, 32))
    output = net(inputs)
    print(net.config)
    print(output)
