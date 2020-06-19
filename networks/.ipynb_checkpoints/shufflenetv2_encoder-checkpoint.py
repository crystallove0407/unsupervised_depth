# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1, depthwise_conv3x3, conv1x1_block, conv3x3_block, ChannelShuffle, SEBlock

class ShuffleUnit(nn.Module):
    """
    ShuffleNetV2 unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    downsample : bool
        Whether do downsample.
    use_se : bool
        Whether to use SE block.
    use_residual : bool
        Whether to use residual connection.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 downsample,
                 use_se,
                 use_residual):
        super(ShuffleUnit, self).__init__()
        self.downsample = downsample
        self.use_se = use_se
        self.use_residual = use_residual
        mid_channels = out_channels // 2

        self.compress_conv1 = conv1x1(
            in_channels=(in_channels if self.downsample else mid_channels),
            out_channels=mid_channels)
        self.compress_bn1 = nn.BatchNorm2d(num_features=mid_channels)
        self.dw_conv2 = depthwise_conv3x3(
            channels=mid_channels,
            stride=(2 if self.downsample else 1))
        self.dw_bn2 = nn.BatchNorm2d(num_features=mid_channels)
        self.expand_conv3 = conv1x1(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.expand_bn3 = nn.BatchNorm2d(num_features=mid_channels)
        if self.use_se:
            self.se = SEBlock(channels=mid_channels)
        if downsample:
            self.dw_conv4 = depthwise_conv3x3(
                channels=in_channels,
                stride=2)
            self.dw_bn4 = nn.BatchNorm2d(num_features=in_channels)
            self.expand_conv5 = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.expand_bn5 = nn.BatchNorm2d(num_features=mid_channels)

        self.activ = nn.ReLU(inplace=True)
        self.c_shuffle = ChannelShuffle(
            channels=out_channels,
            groups=2)

    def forward(self, x):
        if self.downsample:
            y1 = self.dw_conv4(x)
            y1 = self.dw_bn4(y1)
            y1 = self.expand_conv5(y1)
            y1 = self.expand_bn5(y1)
            y1 = self.activ(y1)
            x2 = x
        else:
            y1, x2 = torch.chunk(x, chunks=2, dim=1)
        y2 = self.compress_conv1(x2)
        y2 = self.compress_bn1(y2)
        y2 = self.activ(y2)
        y2 = self.dw_conv2(y2)
        y2 = self.dw_bn2(y2)
        y2 = self.expand_conv3(y2)
        y2 = self.expand_bn3(y2)
        y2 = self.activ(y2)
        if self.use_se:
            y2 = self.se(y2)
        if self.use_residual and not self.downsample:
            y2 = y2 + x2
        x = torch.cat((y1, y2), dim=1)
        x = self.c_shuffle(x)
        return x


class ShuffleInitBlock(nn.Module):
    """
    ShuffleNetV2 specific initial block.

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
        super(ShuffleInitBlock, self).__init__()

        self.conv = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0,
            ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class ShuffleNetV2(nn.Module):
    """
    ShuffleNetV2 model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters:
    ----------
    width_scale : float
        Scale factor for width of layers.
    use_se : bool, default False
        Whether to use SE block.
    use_residual : bool, default False
        Whether to use residual connections.
    in_channels : int, default 3
        Number of input channels.
    """
    def __init__(self,
                 width_scale=1.0,
                 use_se=False,
                 use_residual=False,
                 in_channels=3):
        super(ShuffleNetV2, self).__init__()
        
        init_block_channels = 24
        final_block_channels = 1024
        layers = [4, 8, 4]
        channels_per_layers = [116, 232, 464]
        self.num_ch_enc = np.array([24, 24, 116, 232, 1024])

        channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

        if width_scale != 1.0:
            channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
            if width_scale > 1.5:
                final_block_channels = int(final_block_channels * width_scale)

        self.encoder = nn.Sequential()
        self.encoder.add_module("init_block_conv", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2))
        self.encoder.add_module("init_block_maxpool", nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=0,
            ceil_mode=True))
#         self.encoder.add_module("init_block", ShuffleInitBlock(
#             in_channels=in_channels,
#             out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                downsample = (j == 0)
                stage.add_module("unit{}".format(j + 1), ShuffleUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    downsample=downsample,
                    use_se=use_se,
                    use_residual=use_residual))
                in_channels = out_channels
            self.encoder.add_module("stage{}".format(i + 1), stage)
        self.encoder.add_module("final_block", conv1x1_block(
            in_channels=in_channels,
            out_channels=final_block_channels))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.init_block_conv(x)
        self.features.append(x)
        x = self.encoder.init_block_maxpool(x)
        self.features.append(x)
        x = self.encoder.stage1(x)
        self.features.append(x)
        x = self.encoder.stage2(x)
        self.features.append(x)
        x = self.encoder.stage3(x)
        x = self.encoder.final_block(x)
        self.features.append(x)
        return self.features
