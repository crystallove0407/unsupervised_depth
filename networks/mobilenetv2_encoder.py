import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1, conv1x1_block, conv3x3_block, dwconv3x3_block
from functools import reduce
import numpy as np

class LinearBottleneck(nn.Module):
    """
    So-called 'Linear Bottleneck' layer. It is used as a MobileNetV2 unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    expansion : bool
        Whether do expansion of channels.
    remove_exp_conv : bool
        Whether to remove expansion convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expansion,
                 remove_exp_conv):
        super(LinearBottleneck, self).__init__()
        self.residual = (in_channels == out_channels) and (stride == 1)
        mid_channels = in_channels * 6 if expansion else in_channels
        self.use_exp_conv = (expansion or (not remove_exp_conv))

        if self.use_exp_conv:
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                activation="relu6")
        self.conv2 = dwconv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            activation="relu6")
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        if self.use_exp_conv:
            x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.residual:
            x = x + identity
        return x


class MobileNetV2(nn.Module):
    """
    MobileNetV2 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,' https://arxiv.org/abs/1801.04381.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    remove_exp_conv : bool
        Whether to remove expansion convolution.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 width_scale=1.0,
                 in_channels=3):
        super(MobileNetV2, self).__init__()
        self.num_ch_enc = np.array([16, 24, 32, 96, 320, 1280])
        self.remove_exp_conv = False
        self.init_block_channels = 32
        self.final_block_channels = 1280
        self.layers = [1, 2, 3, 4, 3, 3, 1]
        self.downsample = [0, 1, 1, 1, 0, 1, 0]
        self.channels_per_layers = [16, 24, 32, 64, 96, 160, 320]
        self.channels = reduce(
            lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
            zip(self.channels_per_layers, self.layers, self.downsample),
            [[]])
        if width_scale != 1.0:
            self.channels = [[int(cij * width_scale) for cij in ci] for ci in self.channels]
            self.init_block_channels = int(self.init_block_channels * width_scale)
            if width_scale > 1.0:
                self.final_block_channels = int(self.final_block_channels * width_scale)

        self.encoder = nn.Sequential()
        self.encoder.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=self.init_block_channels,
            stride=2,
            activation="relu6"))
        in_channels = self.init_block_channels
        for i, channels_per_stage in enumerate(self.channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                expansion = (i != 0) or (j != 0)
                stage.add_module("unit{}".format(j + 1), LinearBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expansion=expansion,
                    remove_exp_conv=self.remove_exp_conv))
                in_channels = out_channels
            self.encoder.add_module("stage{}".format(i + 1), stage)
        self.encoder.add_module("final_block", conv1x1_block(
            in_channels=in_channels,
            out_channels=self.final_block_channels,
            activation="relu6"))

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, input_image):
        features = []
        x = (input_image - 0.45) / 0.225
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i != 0 and i != 5:
                features.append(x)
        return features