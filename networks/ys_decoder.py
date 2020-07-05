# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class YSDecoder(nn.Module):
    def __init__(self, num_ch_enc, use_skips=True):
        super(YSDecoder, self).__init__()

        self.num_output_channels = 1
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = range(4)

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        
        # 4
        num_ch_in = self.num_ch_enc[-1]
        num_ch_out = self.num_ch_dec[4]
        self.convs[("upconv", 4, 0)] = ConvBlock(num_ch_in, num_ch_out)
        num_ch_in = self.num_ch_dec[4] + self.num_ch_enc[3]
        num_ch_out = self.num_ch_dec[4]
        self.convs[("upconv", 4, 1)] = ConvBlock(num_ch_in, num_ch_out)
        
        # 3
        num_ch_in = self.num_ch_dec[4]
        num_ch_out = self.num_ch_dec[3]
        self.convs[("upconv", 3, 0)] = ConvBlock(num_ch_in, num_ch_out)
        num_ch_in = self.num_ch_dec[3] + self.num_ch_enc[2]
        num_ch_out = self.num_ch_dec[3]
        self.convs[("upconv", 3, 1)] = ConvBlock(num_ch_in, num_ch_out)
        
        # 2
        num_ch_in = self.num_ch_dec[3]
        num_ch_out = self.num_ch_dec[2]
        self.convs[("upconv", 2, 0)] = ConvBlock(num_ch_in, num_ch_out)
        num_ch_in = self.num_ch_dec[2] + self.num_ch_enc[1] + 1
        num_ch_out = self.num_ch_dec[2]
        self.convs[("upconv", 2, 1)] = ConvBlock(num_ch_in, num_ch_out)
        
        # 1
        num_ch_in = self.num_ch_dec[2]
        num_ch_out = self.num_ch_dec[1]
        self.convs[("upconv", 1, 0)] = ConvBlock(num_ch_in, num_ch_out)
        num_ch_in = self.num_ch_dec[1] + self.num_ch_enc[0] + 1
        num_ch_out = self.num_ch_dec[1]
        self.convs[("upconv", 1, 1)] = ConvBlock(num_ch_in, num_ch_out)
        
        # 0
        num_ch_in = self.num_ch_dec[1]
        num_ch_out = self.num_ch_dec[0]
        self.convs[("upconv", 0, 0)] = ConvBlock(num_ch_in, num_ch_out)
        num_ch_in = self.num_ch_dec[0] + 1
        num_ch_out = self.num_ch_dec[0]
        self.convs[("upconv", 0, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}
        
        # decoder
        x = input_features[-1]
        
        # 4
        x = self.convs[("upconv", 4, 0)](F.interpolate(x, scale_factor=2, mode=self.upsample_mode))
        x = torch.cat([x, input_features[3]], 1)
        x = self.convs[("upconv", 4, 1)](x)
        
        # 3
        x = self.convs[("upconv", 3, 0)](F.interpolate(x, scale_factor=2, mode=self.upsample_mode))
        x = torch.cat([x, input_features[2]], 1)
        x = self.convs[("upconv", 3, 1)](x)
        d_s = self.sigmoid(self.convs[("dispconv", 3)](x))
        
        # 2
        x = self.convs[("upconv", 2, 0)](F.interpolate(x, scale_factor=2, mode=self.upsample_mode))
        x = torch.cat([x, input_features[1], F.interpolate(d_s, scale_factor=2, mode=self.upsample_mode)], 1)
        x = self.convs[("upconv", 2, 1)](x)
        d_x = self.sigmoid(self.convs[("dispconv", 2)](x))
        
        # 1
        x = self.convs[("upconv", 1, 0)](F.interpolate(x, scale_factor=2, mode=self.upsample_mode))
        x = torch.cat([x, input_features[0], F.interpolate(d_x, scale_factor=2, mode=self.upsample_mode)], 1)
        x = self.convs[("upconv", 1, 1)](x)
        d_l = self.sigmoid(self.convs[("dispconv", 1)](x))
        
        # 0
        x = self.convs[("upconv", 0, 0)](F.interpolate(x, scale_factor=2, mode=self.upsample_mode))
        x = torch.cat([x, F.interpolate(d_l, scale_factor=2, mode=self.upsample_mode)], 1)
        x = self.convs[("upconv", 0, 1)](x)
        d_xl = self.sigmoid(self.convs[("dispconv", 0)](x))
        
        
        self.outputs[("disp", 0)] = d_xl
        self.outputs[("disp", 1)] = d_l
        self.outputs[("disp", 2)] = d_x
        self.outputs[("disp", 3)] = d_s

        return self.outputs