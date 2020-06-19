import torch.nn as nn
import torch.nn.init as init
from .common import round_channels, conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block, SEBlock
import numpy as np

class DwsExpSEResUnit(nn.Module):
    """
    Depthwise separable expanded residual unit with SE-block. Here it used as MnasNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the second convolution layer.
    use_kernel3 : bool, default True
        Whether to use 3x3 (instead of 5x5) kernel.
    exp_factor : int, default 1
        Expansion factor for each unit.
    se_factor : int, default 0
        SE reduction factor for each unit.
    use_skip : bool, default True
        Whether to use skip connection.
    activation : str, default 'relu'
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 use_kernel3=True,
                 exp_factor=1,
                 se_factor=0,
                 use_skip=True,
                 activation="relu"):
        super(DwsExpSEResUnit, self).__init__()
        assert (exp_factor >= 1)
        self.residual = (in_channels == out_channels) and (stride == 1) and use_skip
        self.use_exp_conv = exp_factor > 1
        self.use_se = se_factor > 0
        mid_channels = exp_factor * in_channels
        dwconv_block_fn = dwconv3x3_block if use_kernel3 else dwconv5x5_block

        if self.use_exp_conv:
            self.exp_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                activation=activation)
        self.dw_conv = dwconv_block_fn(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            activation=activation)
        if self.use_se:
            self.se = SEBlock(
                channels=mid_channels,
                reduction=(exp_factor * se_factor),
                round_mid=False,
                mid_activation=activation)
        self.pw_conv = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        if self.use_exp_conv:
            x = self.exp_conv(x)
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        if self.residual:
            x = x + identity
        return x


class MnasInitBlock(nn.Module):
    """
    MnasNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    use_skip : bool
        Whether to use skip connection in the second block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 use_skip):
        super(MnasInitBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=2)
        self.conv2 = DwsExpSEResUnit(
            in_channels=mid_channels,
            out_channels=out_channels,
            use_skip=use_skip)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MnasFinalBlock(nn.Module):
    """
    MnasNet specific final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    use_skip : bool
        Whether to use skip connection in the second block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 use_skip):
        super(MnasFinalBlock, self).__init__()
        self.conv1 = DwsExpSEResUnit(
            in_channels=in_channels,
            out_channels=mid_channels,
            exp_factor=6,
            use_skip=use_skip)
        self.conv2 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MnasNet(nn.Module):
    """
    MnasNet model from 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,'
    https://arxiv.org/abs/1807.11626.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : list of 2 int
        Number of output channels for the initial unit.
    final_block_channels : list of 2 int
        Number of output channels for the final block of the feature extractor.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 5x5) kernel for each unit.
    exp_factors : list of list of int
        Expansion factor for each unit.
    se_factors : list of list of int
        SE reduction factor for each unit.
    init_block_use_skip : bool
        Whether to use skip connection in the initial unit.
    final_block_use_skip : bool
        Whether to use skip connection in the final block of the feature extractor.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 version="a1",
                 width_scale=1.0,
                 in_channels=3):
        super(MnasNet, self).__init__()
        self.num_ch_enc = np.array([16, 24, 40, 112, 160, 1280])
        if version == "b1":
            init_block_channels = [32, 16]
            final_block_channels = [320, 1280]
            channels = [[24, 24, 24], [40, 40, 40], [80, 80, 80, 96, 96], [192, 192, 192, 192]]
            kernels3 = [[1, 1, 1], [0, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 0]]
            exp_factors = [[3, 3, 3], [3, 3, 3], [6, 6, 6, 6, 6], [6, 6, 6, 6]]
            se_factors = [[0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0]]
            init_block_use_skip = False
            final_block_use_skip = False
        elif version == "a1":
            init_block_channels = [32, 16]
            final_block_channels = [320, 1280]
            channels = [[24, 24], [40, 40, 40], [80, 80, 80, 80, 112, 112], [160, 160, 160]]
            kernels3 = [[1, 1], [0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0]]
            exp_factors = [[6, 6], [3, 3, 3], [6, 6, 6, 6, 6, 6], [6, 6, 6]]
            se_factors = [[0, 0], [4, 4, 4], [0, 0, 0, 0, 4, 4], [4, 4, 4]]
            init_block_use_skip = False
            final_block_use_skip = True
        elif version == "small":
            init_block_channels = [8, 8]
            final_block_channels = [144, 1280]
            channels = [[16], [16, 16], [32, 32, 32, 32, 32, 32, 32], [88, 88, 88]]
            kernels3 = [[1], [1, 1], [0, 0, 0, 0, 1, 1, 1], [0, 0, 0]]
            exp_factors = [[3], [6, 6], [6, 6, 6, 6, 6, 6, 6], [6, 6, 6]]
            se_factors = [[0], [0, 0], [4, 4, 4, 4, 4, 4, 4], [4, 4, 4]]
            init_block_use_skip = True
            final_block_use_skip = True
        else:
            raise ValueError("Unsupported MnasNet version {}".format(version))
        
        if width_scale != 1.0:
            channels = [[round_channels(cij * width_scale) for cij in ci] for ci in channels]
            init_block_channels = round_channels(init_block_channels * width_scale)
        
        self.encoder = nn.Sequential()
        self.encoder.add_module("init_block", MnasInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels[1],
            mid_channels=init_block_channels[0],
            use_skip=init_block_use_skip))
        in_channels = init_block_channels[1]
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) else 1
                use_kernel3 = kernels3[i][j] == 1
                exp_factor = exp_factors[i][j]
                se_factor = se_factors[i][j]
                stage.add_module("unit{}".format(j + 1), DwsExpSEResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    use_kernel3=use_kernel3,
                    exp_factor=exp_factor,
                    se_factor=se_factor))
                in_channels = out_channels
            self.encoder.add_module("stage{}".format(i + 1), stage)
        self.encoder.add_module("final_block", MnasFinalBlock(
            in_channels=in_channels,
            out_channels=final_block_channels[1],
            mid_channels=final_block_channels[0],
            use_skip=final_block_use_skip))

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
            self.features.append(x)
        return self.features
