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
from .common import conv1x1, conv3x3, conv1x1_block, conv3x3_block, conv5x5_block, dwsconv3x3_block, dwsconv5x5_block, DwsConvBlock
import torch.nn.functional as F

    
class OursDecoder(nn.Module):

    def __init__(self, num_ch_enc, use_skips=True, bn=False, dw=False, pw=False, oneLayer=False, add=False, add_input=False):
        super(OursDecoder, self).__init__()
        self.num_output_channels = 1
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = range(4)
        self.convs = OrderedDict()
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([512, 256, 128, 64, 32, 16])
        self.ELU = (lambda: nn.ELU(inplace=True))
        
        self.bn = bn
        self.dw = dw
        self.pw = pw
        self.oneLayer = oneLayer
        self.add = add
        self.add_input = add_input
        
        if self.add:
            self.in_channels = [self.num_ch_enc[-1], 
                               512, 
                               256,
                               128,
                               64,
                               32]
        else:
            self.in_channels = [self.num_ch_enc[-1], 
                               512, 
                               256+self.num_ch_enc[3],
                               128+self.num_ch_enc[2],
                               64+self.num_ch_enc[1],
                               32+self.num_ch_enc[0]]
            
        if self.add_input and not self.add:
            self.in_channels[5] += 3
            
        
        for i in range(5):
            # upconv_0
            num_ch_in = self.in_channels[i]
            num_ch_out = self.num_ch_dec[i]
            if self.dw:
                self.convs[("conv", i, 0)] = DwsConvBlock(in_channels=num_ch_in,
                                                out_channels=num_ch_out,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                use_bn=self.bn,
                                                dw_activation=self.ELU
                                                pw_activation=self.ELU)
            else:
                self.convs[("conv", i, 0)] = conv3x3_block(num_ch_in, 
                                                           num_ch_out, 
                                                           use_bn=self.bn, 
                                                           activation=self.ELU)
            
            if not (self.oneLayer and i < 4):
                # upconv_1
                num_ch_in = self.num_ch_dec[i]
                num_ch_out = self.num_ch_dec[i]
                if self.dw:
                    self.convs[("conv", i, 1)] = DwsConvBlock(in_channels=num_ch_in,
                                                    out_channels=num_ch_out,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    use_bn=self.bn,
                                                    dw_activation=self.ELU
                                                    pw_activation=self.ELU)
                else:
                    self.convs[("conv", i, 1)] = conv3x3_block(num_ch_in, 
                                                               num_ch_out, 
                                                               use_bn=self.bn, 
                                                               activation=self.ELU)
        
            
        
        if self.pw:
            self.convs[("dispconv", 3)] = conv1x1(128, self.num_output_channels)
            self.convs[("dispconv", 2)] = conv1x1(64, self.num_output_channels)
            self.convs[("dispconv", 1)] = conv1x1(32, self.num_output_channels)
            self.convs[("dispconv", 0)] = conv1x1(16, self.num_output_channels)
        else:
            self.convs[("dispconv", 3)] = conv3x3(128, self.num_output_channels)
            self.convs[("dispconv", 2)] = conv3x3(64, self.num_output_channels)
            self.convs[("dispconv", 1)] = conv3x3(32, self.num_output_channels)
            self.convs[("dispconv", 0)] = conv3x3(16, self.num_output_channels)
            

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, input_features, input_image=None):
        self.outputs = {}
        x = input_features[-1]
        
        for i in range(5):
            x = self.convs[("conv", i, 0)](x)
            x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
            if i < 4 or self.add_input:
                skip_connect = input_features[3-i] if i < 4 else input_image
                if self.add:
                    x += skip_connect
                else:
                    x = torch.cat((x, skip_connect), 1)
                    
            if not (self.oneLayer and i < 4):
                x = self.conv[("conv", i, 1)](x)
            
            if i > 0:
                self.outputs[("disp", 4-i)] = self.sigmoid(self.convs[("dispconv", 4-i)](x))
        
        return self.outputs
    

        
