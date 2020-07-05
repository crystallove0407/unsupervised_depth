from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from layers import *
from .common import dwsconv3x3_block, conv1x1_block, conv1x1

class mobilenetv2_bottlenet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion_factor=6,
                 alpha=1.0,
                 bn_eps=1e-5,
                 dw_activation=(lambda: nn.ReLU6(inplace=True)),
                 pw_activation=(lambda: nn.ReLU6(inplace=True))):
        super().__init__()
        depthwise_conv_filters = self._make_divisible(in_channels * expansion_factor)
        pointwise_conv_filters = self._make_divisible(out_channels * alpha)
        
        self.pw_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=depthwise_conv_filters,
            activation=pw_activation)
        self.dpw_conv = dwsconv3x3_block(
            in_channels=depthwise_conv_filters,
            out_channels=pointwise_conv_filters,
            dw_activation=dw_activation,
            pw_activation=None)
        
    def _make_divisible(self, v, divisor=8, min_value=8):
        if min_value is None:
            min_value = divisor

        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v
        
    def forward(self, x):
        x = self.pw_conv(x)
        x = self.dpw_conv(x)
        
        return x

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


class MJDecoder(nn.Module):
    def __init__(self, num_ch_enc, use_skips=True):
        super().__init__()

        self.num_output_channels = 1
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = range(4)

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 24, 32, 64, 128])
        # decoder
        self.convs = OrderedDict()

        
        self.convs[("bottlenet", 3, 0)] = mobilenetv2_bottlenet(self.num_ch_enc[-1] // 16, self.num_ch_dec[3])
        self.convs[("bottlenet", 3, 1)] = mobilenetv2_bottlenet(self.num_ch_dec[3], self.num_ch_dec[3])
        self.convs[("bottlenet", 3, 2)] = mobilenetv2_bottlenet(self.num_ch_dec[3], self.num_ch_dec[3])
        self.convs[("bottlenet", 3, 3)] = mobilenetv2_bottlenet(self.num_ch_dec[3], self.num_ch_dec[3])
        
        self.convs[("bottlenet", 2, 0)] = mobilenetv2_bottlenet((self.num_ch_dec[3] + self.num_ch_enc[2]) // 4, self.num_ch_dec[2])
        self.convs[("bottlenet", 2, 1)] = mobilenetv2_bottlenet(self.num_ch_dec[2], self.num_ch_dec[2])
        self.convs[("bottlenet", 2, 2)] = mobilenetv2_bottlenet(self.num_ch_dec[2], self.num_ch_dec[2])
        

        self.convs[("bottlenet", 1, 0)] = mobilenetv2_bottlenet((self.num_ch_dec[2] + self.num_ch_enc[1]) // 4, self.num_ch_dec[1])
        self.convs[("bottlenet", 1, 1)] = mobilenetv2_bottlenet(self.num_ch_dec[1], self.num_ch_dec[1])
        self.convs[("bottlenet", 1, 2)] = mobilenetv2_bottlenet(self.num_ch_dec[1], self.num_ch_dec[1])
        
        

        self.convs[("bottlenet", 0, 0)] = mobilenetv2_bottlenet((self.num_ch_dec[1] + self.num_ch_enc[0]) // 4, self.num_ch_dec[0])
        self.convs[("bottlenet", 0, 1)] = mobilenetv2_bottlenet(self.num_ch_dec[0], self.num_ch_dec[0])
            
        



        for s in self.scales:
#             self.convs[("dispconv", s)] = conv1x1(self.num_ch_dec[s], self.num_output_channels)
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
        self.convs[("upconv", 3)] = DepthToSpace(4)
        self.convs[("upconv", 2)] = DepthToSpace(2)
        self.convs[("upconv", 1)] = DepthToSpace(2)
        self.convs[("upconv", 0)] = DepthToSpace(2)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}
        # decoder
        x = input_features[-1]
#         x = F.pixel_shuffle(x, 4)
#         x = self.convs[("upconv", 4)](x)
        
        x = self.convs[("upconv", 3)](x)
        x = self.convs[("bottlenet", 3, 0)](x)
        x = self.convs[("bottlenet", 3, 1)](x)
        x = self.convs[("bottlenet", 3, 2)](x)
        x = self.convs[("bottlenet", 3, 3)](x)
        d_s = x
        
        x = torch.cat([x, input_features[2]], 1)
        x = self.convs[("upconv", 2)](x)
#         x = F.pixel_shuffle(x, 2)
        x = self.convs[("bottlenet", 2, 0)](x)
        x = self.convs[("bottlenet", 2, 1)](x)
        x = self.convs[("bottlenet", 2, 2)](x)
        d_x = x
        
        x = torch.cat([x, input_features[1]], 1)
        x = self.convs[("upconv", 1)](x)
#         x = F.pixel_shuffle(x, 2)
        x = self.convs[("bottlenet", 1, 0)](x)
        x = self.convs[("bottlenet", 1, 1)](x)
        x = self.convs[("bottlenet", 1, 2)](x)
        d_l = x
        
        x = torch.cat([x, input_features[0]], 1)
        x = self.convs[("upconv", 0)](x)
#         x = F.pixel_shuffle(x, 2)
        x = self.convs[("bottlenet", 0, 0)](x)
        x = self.convs[("bottlenet", 0, 1)](x)
        d_xl = x
        
        d_s = self.sigmoid(self.convs[("dispconv", 3)](d_s))
        d_x = self.sigmoid(self.convs[("dispconv", 2)](d_x))
        d_l = self.sigmoid(self.convs[("dispconv", 1)](d_l))
        d_xl = self.sigmoid(self.convs[("dispconv", 0)](d_xl))
        
        self.outputs[("disp", 0)] = d_xl
        self.outputs[("disp", 1)] = d_l
        self.outputs[("disp", 2)] = d_x
        self.outputs[("disp", 3)] = d_s

        return self.outputs
        