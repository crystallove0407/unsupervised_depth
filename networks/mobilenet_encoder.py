import os
import torch.nn as nn
from .common import conv3x3_block, dwsconv3x3_block


class MobileNet(nn.Module):
    """
    MobileNet model from 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.
    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    first_stage_stride : bool
        Whether stride is used at the first stage.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 in_channels=3):
        super(MobileNet, self).__init__()
        self.num_ch_enc = [64, 128, 256, 512, 1024]
        self.channels = [[32], 
                         [64], 
                         [128, 128], 
                         [256, 256], 
                         [512, 512, 512, 512, 512, 512], 
                         [1024, 1024]]
        self.first_stage_stride = False
        self.encoder = nn.Sequential()
        init_block_channels = self.channels[0][0]
        self.encoder.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(self.channels[1:]):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and ((i != 0) or self.first_stage_stride) else 1
                stage.add_module("unit{}".format(j + 1), dwsconv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride))
                in_channels = out_channels
            self.encoder.add_module("stage{}".format(i + 1), stage)


        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if 'dw_conv.conv' in name:
                nn.init.kaiming_normal_(module.weight, mode='fan_in')
            elif name == 'init_block.conv' or 'pw_conv.conv' in name:
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
            elif 'bn' in name:
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif 'output' in name:
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = []
        x = (input_image - 0.45) / 0.225
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i != 0:
                features.append(x)
        return features