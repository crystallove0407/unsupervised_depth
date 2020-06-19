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
from .common import conv1x1, conv3x3, conv1x1_block, conv3x3_block, conv5x5_block, dwsconv3x3_block, dwsconv5x5_block
import torch.nn.functional as F

    
class NNDecoder(nn.Module):

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, upsample_mode='nearest', dw=False, pw=False):
        super(NNDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = upsample_mode
        self.scales = scales
        self.convs = OrderedDict()
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.upconv = upconv
        

        if dw:
            self.convs["conv1"] = dwsconv3x3_block(512, 256)
            self.convs["conv2"] = dwsconv3x3_block(256+self.num_ch_enc[3], 128)
            self.convs["conv3"] = dwsconv3x3_block(128+self.num_ch_enc[2], 64)
            self.convs["conv4"] = dwsconv3x3_block(64+self.num_ch_enc[1], 32)
            self.convs["conv5"] = dwsconv3x3_block(32 +self.num_ch_enc[0], 16)
        else:
            self.convs["conv1"] = conv3x3_block(512, 256)
            self.convs["conv2"] = conv3x3_block(256+self.num_ch_enc[3], 128)
            self.convs["conv3"] = conv3x3_block(128+self.num_ch_enc[2], 64)
            self.convs["conv4"] = conv3x3_block(64+self.num_ch_enc[1], 32)
            self.convs["conv5"] = conv3x3_block(32 +self.num_ch_enc[0], 16)
        
        
        if pw == 3:
            self.convs[("dispconv", 3)] = conv3x3(128, self.num_output_channels)
            self.convs[("dispconv", 2)] = conv3x3(64, self.num_output_channels)
            self.convs[("dispconv", 1)] = conv3x3(32, self.num_output_channels)
            self.convs[("dispconv", 0)] = conv3x3(16, self.num_output_channels)
        elif pw == 1:
            self.convs[("dispconv", 3)] = conv1x1(128, self.num_output_channels)
            self.convs[("dispconv", 2)] = conv1x1(64, self.num_output_channels)
            self.convs[("dispconv", 1)] = conv1x1(32, self.num_output_channels)
            self.convs[("dispconv", 0)] = conv1x1(16, self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, input_features):
        self.outputs = {}
        

        x = input_features[-1]
        x = self.convs["conv1"](x)
        x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)

        x = torch.cat((x, input_features[3]), 1)
        x = self.convs["conv2"](x)
        x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
        self.outputs[("disp", 3)] = self.sigmoid(self.convs[("dispconv", 3)](x))

        x = torch.cat((x, input_features[2]), 1)
        x = self.convs["conv3"](x)
        x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
        self.outputs[("disp", 2)] = self.sigmoid(self.convs[("dispconv", 2)](x))

        x = torch.cat((x, input_features[1]), 1)
        x = self.convs["conv4"](x)
        x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
        self.outputs[("disp", 1)] = self.sigmoid(self.convs[("dispconv", 1)](x))

        x = torch.cat((x, input_features[0]), 1)
        x = self.convs["conv5"](x)
        x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
        self.outputs[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](x))
        
        

        return self.outputs
    

        
