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


class MonoDecoder(nn.Module):
    def __init__(self, num_ch_enc, use_skips=True):
        super(MonoDecoder, self).__init__()

        self.num_output_channels = 1
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = range(4)

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}
        
        # decoder
        x = input_features[-1]
        
        x = self.convs[("upconv", 4, 0)](x)
        x = [F.interpolate(x, scale_factor=2, mode=self.upsample_mode)]
        x += [input_features[3]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 4, 1)](x)
        
        
        x = self.convs[("upconv", 3, 0)](x)
        x = [F.interpolate(x, scale_factor=2, mode=self.upsample_mode)]
        x += [input_features[2]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 3, 1)](x)
        d_s = x
        
        x = self.convs[("upconv", 2, 0)](x)
        x = [F.interpolate(x, scale_factor=2, mode=self.upsample_mode)]
        x += [input_features[1]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 2, 1)](x)
        d_x = x
        
        x = self.convs[("upconv", 1, 0)](x)
        x = [F.interpolate(x, scale_factor=2, mode=self.upsample_mode)]
        x += [input_features[0]]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 1, 1)](x)
        d_l = x
        
        x = self.convs[("upconv", 0, 0)](x)
        x = [F.interpolate(x, scale_factor=2, mode=self.upsample_mode)]
        x = torch.cat(x, 1)
        x = self.convs[("upconv", 0, 1)](x)
        d_xl = x
        
        d_s  = self.sigmoid(self.convs[("dispconv", 3)](d_s))
        d_x  = self.sigmoid(self.convs[("dispconv", 2)](d_x))
        d_l  = self.sigmoid(self.convs[("dispconv", 1)](d_l))
        d_xl = self.sigmoid(self.convs[("dispconv", 0)](d_xl))

        self.outputs[("disp", 3)] = d_xl
        self.outputs[("disp", 2)] = d_l
        self.outputs[("disp", 1)] = d_x
        self.outputs[("disp", 0)] = d_s

        return self.outputs