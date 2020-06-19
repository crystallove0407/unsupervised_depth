import torch
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1_block, conv3x3_block, Concurrent
from functools import reduce
import numpy as np

class PeleeBranch1(nn.Module):
    """
    PeleeNet branch type 1 block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the second convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 stride=1):
        super(PeleeBranch1, self).__init__()
        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            stride=stride)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PeleeBranch2(nn.Module):
    """
    PeleeNet branch type 2 block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels):
        super(PeleeBranch2, self).__init__()
        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels)
        self.conv3 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class StemBlock(nn.Module):
    """
    PeleeNet stem block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(StemBlock, self).__init__()
        mid1_channels = out_channels // 2
        mid2_channels = out_channels * 2

        self.first_conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2)

        self.branches = Concurrent()
        self.branches.add_module("branch1", PeleeBranch1(
            in_channels=out_channels,
            out_channels=out_channels,
            mid_channels=mid1_channels,
            stride=2))
        self.branches.add_module("branch2", nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=0))

        self.last_conv = conv1x1_block(
            in_channels=mid2_channels,
            out_channels=out_channels)

    def forward(self, x0):
        x1 = self.first_conv(x0)
        x2 = self.branches(x1)
        x3 = self.last_conv(x2)
        return x1, x3


class DenseBlock(nn.Module):
    """
    PeleeNet dense block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bottleneck_size : int
        Bottleneck width.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bottleneck_size):
        super(DenseBlock, self).__init__()
        inc_channels = (out_channels - in_channels) // 2
        mid_channels = inc_channels * bottleneck_size

        self.branch1 = PeleeBranch1(
            in_channels=in_channels,
            out_channels=inc_channels,
            mid_channels=mid_channels)
        self.branch2 = PeleeBranch2(
            in_channels=in_channels,
            out_channels=inc_channels,
            mid_channels=mid_channels)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat((x, x1, x2), dim=1)
        return x


class TransitionBlock(nn.Module):
    """
    PeleeNet's transition block, like in DensNet, but with ordinary convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(TransitionBlock, self).__init__()
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels)
        self.pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
            padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class PeleeNet(nn.Module):
    """
    PeleeNet model from 'Pelee: A Real-Time Object Detection System on Mobile Devices,'
    https://arxiv.org/abs/1804.06882.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck_sizes : list of int
        Bottleneck sizes for each stage.
    dropout_rate : float, default 0.5
        Parameter of Dropout layer. Faction of the input units to drop.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 dropout_rate=0.5,
                 in_channels=3):
        super(PeleeNet, self).__init__()
        
        self.num_ch_enc = np.array([32, 128, 256, 512, 704, 704])
        self.init_block_channels = 32
        self.growth_rate = 32
        self.layers = [3, 4, 8, 6]
        self.bottleneck_sizes = [1, 2, 4, 4]

        self.channels = reduce(
            lambda xi, yi: xi + [reduce(
                lambda xj, yj: xj + [xj[-1] + yj],
                [self.growth_rate] * yi,
                [xi[-1][-1]])[1:]],
            self.layers,
            [[self.init_block_channels]])[1:]
        
        
        self.encoder = nn.Sequential()
        self.encoder.add_module("init_block", StemBlock(
            in_channels=in_channels,
            out_channels=self.init_block_channels))
        in_channels = self.init_block_channels
        for i, channels_per_stage in enumerate(self.channels):
            bottleneck_size = self.bottleneck_sizes[i]
            stage = nn.Sequential()
            if i != 0:
                stage.add_module("trans{}".format(i + 1), TransitionBlock(
                    in_channels=in_channels,
                    out_channels=in_channels))
            for j, out_channels in enumerate(channels_per_stage):
                stage.add_module("unit{}".format(j + 1), DenseBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    bottleneck_size=bottleneck_size))
                in_channels = out_channels
            self.encoder.add_module("stage{}".format(i + 1), stage)
        self.encoder.add_module("final_block", conv1x1_block(
            in_channels=in_channels,
            out_channels=in_channels))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        for i, layer in enumerate(self.encoder):
            if i == 0:
                x1, x = layer(x)
                self.features.append(x1)
            else:
                x = layer(x)
                self.features.append(x)
        return self.features
