import os
import torch.nn as nn
import torch.nn.init as init
from .common import round_channels, conv1x1, conv1x1_block, conv3x3_block, dwconv3x3_block, \
                    dwconv5x5_block, SEBlock, HSwish
import numpy as np

class MobileNetV3Unit(nn.Module):
    """
    MobileNetV3 unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    self.exp_channels : int
        Number of middle (expanded) channels.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    activation : str
        Activation function or name of activation function.
    use_se : bool
        Whether to use SE-module.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 exp_channels,
                 stride,
                 use_kernel3,
                 activation,
                 use_se):
        super(MobileNetV3Unit, self).__init__()
        assert (exp_channels >= out_channels)
        self.residual = (in_channels == out_channels) and (stride == 1)
        self.use_se = use_se
        self.use_exp_conv = exp_channels != out_channels
        mid_channels = exp_channels

        if self.use_exp_conv:
            self.exp_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                activation=activation)
        if use_kernel3:
            self.conv1 = dwconv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                activation=activation)
        else:
            self.conv1 = dwconv5x5_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                activation=activation)
        if self.use_se:
            self.se = SEBlock(
                channels=mid_channels,
                reduction=4,
                round_mid=True,
                out_activation="hsigmoid")
        self.conv2 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        if self.use_exp_conv:
            x = self.exp_conv(x)
        x = self.conv1(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv2(x)
        if self.residual:
            x = x + identity
        return x


class MobileNetV3FinalBlock(nn.Module):
    """
    MobileNetV3 final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_se : bool
        Whether to use SE-module.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_se):
        super(MobileNetV3FinalBlock, self).__init__()
        self.use_se = use_se

        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation="hswish")
        if self.use_se:
            self.se = SEBlock(
                channels=out_channels,
                reduction=4,
                round_mid=True,
                out_activation="hsigmoid")

    def forward(self, x):
        x = self.conv(x)
        if self.use_se:
            x = self.se(x)
        return x


class MobileNetV3Classifier(nn.Module):
    """
    MobileNetV3 classifier.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 dropout_rate):
        super(MobileNetV3Classifier, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)

        self.conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.activ = HSwish(inplace=True)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels,
            bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activ(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        return x


class MobileNetV3(nn.Module):
    """
    MobileNetV3 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    exp_channels : list of list of int
        Number of middle (expanded) channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    classifier_mid_channels : int
        Number of middle channels for classifier.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 5x5) kernel for each unit.
    use_relu : list of list of int/bool
        Using ReLU activation flag for each unit.
    use_se : list of list of int/bool
        Using SE-block flag for each unit.
    first_stride : bool
        Whether to use stride for the first stage.
    final_use_se : bool
        Whether to use SE-module in the final block.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 version='large',
                 width_scale=1.0,
                 in_channels=3):
        super(MobileNetV3, self).__init__()
        self.num_ch_enc = np.array([16, 16, 24, 48, 96, 576])
        
        if version == "small":
            self.init_block_channels = 16
            self.channels = [[16], [24, 24], [40, 40, 40, 48, 48], [96, 96, 96]]
            self.exp_channels = [[16], [72, 88], [96, 240, 240, 120, 144], [288, 576, 576]]
            self.kernels3 = [[1], [1, 1], [0, 0, 0, 0, 0], [0, 0, 0]]
            self.use_relu = [[1], [1, 1], [0, 0, 0, 0, 0], [0, 0, 0]]
            self.use_se = [[1], [0, 0], [1, 1, 1, 1, 1], [1, 1, 1]]
            self.first_stride = True
            self.final_block_channels = 576
        elif version == "large":
            self.init_block_channels = 16
            self.channels = [[16], [24, 24], [40, 40, 40], [80, 80, 80, 80, 112, 112], [160, 160, 160]]
            self.exp_channels = [[16], [64, 72], [72, 120, 120], [240, 200, 184, 184, 480, 672], [672, 960, 960]]
            self.kernels3 = [[1], [1, 1], [0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0]]
            self.use_relu = [[1], [1, 1], [1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0]]
            self.use_se = [[0], [0, 0], [1, 1, 1], [0, 0, 0, 0, 1, 1], [1, 1, 1]]
            self.first_stride = False
            self.final_block_channels = 960
        else:
            raise ValueError("Unsupported MobileNetV3 version {}".format(version))

        self.final_use_se = False
        self.classifier_mid_channels = 1280

        if width_scale != 1.0:
            self.channels = [[round_channels(cij * width_scale) for cij in ci] for ci in self.channels]
            self.exp_channels = [[round_channels(cij * width_scale) for cij in ci] for ci in self.exp_channels]
            self.init_block_channels = round_channels(self.init_block_channels * width_scale)
            if width_scale > 1.0:
                self.final_block_channels = round_channels(self.final_block_channels * width_scale)
        
        

        self.encoder = nn.Sequential()
        self.encoder.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=self.init_block_channels,
            stride=2,
            activation="hswish"))
        in_channels = self.init_block_channels
        for i, channels_per_stage in enumerate(self.channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                self.exp_channels_ij = self.exp_channels[i][j]
                stride = 2 if (j == 0) and ((i != 0) or self.first_stride) else 1
                use_kernel3 = self.kernels3[i][j] == 1
                activation = "relu" if self.use_relu[i][j] == 1 else "hswish"
                use_se_flag = self.use_se[i][j] == 1
                stage.add_module("unit{}".format(j + 1), MobileNetV3Unit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    exp_channels=self.exp_channels_ij,
                    use_kernel3=use_kernel3,
                    stride=stride,
                    activation=activation,
                    use_se=use_se_flag))
                in_channels = out_channels
            self.encoder.add_module("stage{}".format(i + 1), stage)
        self.encoder.add_module("final_block", MobileNetV3FinalBlock(
            in_channels=in_channels,
            out_channels=self.final_block_channels,
            use_se=self.final_use_se))

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
            x = layer(x)
            if i != 0 and i != 5:
                self.features.append(x)
        return self.features