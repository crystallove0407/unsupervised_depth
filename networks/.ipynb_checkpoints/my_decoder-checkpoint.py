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

    
class MYDecoder(nn.Module):

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, kernel_size=3, dw=False, pw=True, moreConv=0, moreFeature=False, concatDepth=False, doubleConv=0, firstConv=False, conv11=False):
        super(MYDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.convs = OrderedDict()
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        
        # test
        self.moreFeature = moreFeature
        self.moreConv = moreConv
        self.concatDepth = concatDepth
        self.doubleConv = doubleConv
        self.firstConv = firstConv
        self.conv11 = conv11
        
        if self.firstConv:
            self.convs["conv0"] = conv3x3_block(512, 256)
        if kernel_size == 3 and dw == False and self.doubleConv == 1:
            self.convs["doubleConv1"] = conv1x1_block(256, 256)
            self.convs["doubleConv2"] = conv1x1_block(128, 128)
            self.convs["doubleConv3"] = conv1x1_block(64, 64)
            self.convs["doubleConv4"] = conv1x1_block(32, 32)
            self.convs["doubleConv5"] = conv1x1_block(16, 16)
        if kernel_size == 3 and dw == False and self.doubleConv == 3:
            self.convs["doubleConv1"] = conv3x3_block(256, 256)
            self.convs["doubleConv2"] = conv3x3_block(128, 128)
            self.convs["doubleConv3"] = conv3x3_block(64, 64)
            self.convs["doubleConv4"] = conv3x3_block(32, 32)
            self.convs["doubleConv5"] = conv3x3_block(16, 16)
        if kernel_size == 3:
            if dw:
                self.convs["conv1"] = dwsconv3x3_block(512+self.num_ch_enc[3], 256)
                self.convs["conv2"] = dwsconv3x3_block(256+self.num_ch_enc[2], 128)
                self.convs["conv3"] = dwsconv3x3_block(128+self.num_ch_enc[1], 64)
                self.convs["conv4"] = dwsconv3x3_block(64 +self.num_ch_enc[0], 32)
                self.convs["conv5"] = dwsconv3x3_block(32, 16)
            else:
                if self.firstConv:
                    self.convs["conv1"] = conv3x3_block(256+self.num_ch_enc[3], 256)   
                else:
                    self.convs["conv1"] = conv3x3_block(512+self.num_ch_enc[3], 256)
                if self.conv11:
                    self.convs["conv11"] = conv3x3_block(256, 256)
                
                self.convs["conv2"] = conv3x3_block(256+self.num_ch_enc[2], 128)
                if self.concatDepth:
                    self.convs["conv3"] = conv3x3_block(128+self.num_ch_enc[1]+1, 64)
                    self.convs["conv4"] = conv3x3_block(64 +self.num_ch_enc[0]+1, 32)
                    self.convs["conv5"] = conv3x3_block(32+1, 16)
                else:
                    self.convs["conv3"] = conv3x3_block(128+self.num_ch_enc[1], 64)
                    self.convs["conv4"] = conv3x3_block(64 +self.num_ch_enc[0], 32)
                    self.convs["conv5"] = conv3x3_block(32, 16)
                if self.moreConv == 1:
                    self.convs["conv6"] = conv3x3_block(16, 16)
                elif self.moreConv == 2:
                    self.convs["conv6"] = conv3x3_block(16, 16)
                    self.convs["conv7"] = conv3x3_block(16, 16)

        elif kernel_size == 5:
            if dw:
                self.convs["conv1"] = dwsconv5x5_block(512+self.num_ch_enc[3], 256)
                self.convs["conv2"] = dwsconv5x5_block(256+self.num_ch_enc[2], 128)
                self.convs["conv3"] = dwsconv5x5_block(128+self.num_ch_enc[1], 64)
                self.convs["conv4"] = dwsconv5x5_block(64 +self.num_ch_enc[0], 32)
                self.convs["conv5"] = dwsconv5x5_block(32, 16)
            else:
                self.convs["conv1"] = conv5x5_block(512+self.num_ch_enc[3], 256)
                self.convs["conv2"] = conv5x5_block(256+self.num_ch_enc[2], 128)
                self.convs["conv3"] = conv5x5_block(128+self.num_ch_enc[1], 64)
                self.convs["conv4"] = conv5x5_block(64 +self.num_ch_enc[0], 32)
                self.convs["conv5"] = conv5x5_block(32, 16)
        elif kernel_size == 53:
            self.convs["conv1"] = conv5x5_block(512+self.num_ch_enc[3], 256)
            self.convs["conv2"] = conv5x5_block(256+self.num_ch_enc[2], 128)
            self.convs["conv3"] = conv3x3_block(128+self.num_ch_enc[1], 64)
            self.convs["conv4"] = conv3x3_block(64 +self.num_ch_enc[0], 32)
            self.convs["conv5"] = conv3x3_block(32, 16)
        elif kernel_size == 35:
            self.convs["conv1"] = conv3x3_block(512+self.num_ch_enc[3], 256)
            self.convs["conv2"] = conv3x3_block(256+self.num_ch_enc[2], 128)
            self.convs["conv3"] = conv3x3_block(128+self.num_ch_enc[1], 64)
            self.convs["conv4"] = conv5x5_block(64 +self.num_ch_enc[0], 32)
            self.convs["conv5"] = conv5x5_block(32, 16)
            if self.moreFeature:
                self.convs["conv45"] = conv5x5_block(32+self.num_ch_enc[0], 32)
        
        if pw == False:
            self.convs[("dispconv", 3)] = conv3x3(128, self.num_output_channels)
            self.convs[("dispconv", 2)] = conv3x3(64, self.num_output_channels)
            self.convs[("dispconv", 1)] = conv3x3(32, self.num_output_channels)
            self.convs[("dispconv", 0)] = conv3x3(16, self.num_output_channels)
        elif pw == True:
            self.convs[("dispconv", 3)] = conv1x1(128, self.num_output_channels)
            self.convs[("dispconv", 2)] = conv1x1(64, self.num_output_channels)
            self.convs[("dispconv", 1)] = conv1x1(32, self.num_output_channels)
            self.convs[("dispconv", 0)] = conv1x1(16, self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, input_features):
        self.outputs = {}
        

        x = input_features[-1]
        if self.firstConv:
            x = self.convs["conv0"](x)
        
        x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat((x, input_features[3]), 1)
        x = self.convs["conv1"](x)
        if self.conv11:
            x = self.convs["conv11"](x)
        if self.doubleConv > 0:
            x = self.convs["doubleConv1"](x)
        
        
        x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
        x = torch.cat((x, input_features[2]), 1)
        x = self.convs["conv2"](x)
        if self.doubleConv > 0:
            x = self.convs["doubleConv2"](x)
        self.outputs[("disp", 3)] = self.sigmoid(self.convs[("dispconv", 3)](x))
        
        
        x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
        if self.concatDepth:
            depth = F.interpolate(self.outputs[("disp", 3)], scale_factor=2, mode=self.upsample_mode)
            x = torch.cat((x, input_features[1], depth), 1)
        else:
            x = torch.cat((x, input_features[1]), 1)
        x = self.convs["conv3"](x)
        if self.doubleConv > 0:
            x = self.convs["doubleConv3"](x)
        self.outputs[("disp", 2)] = self.sigmoid(self.convs[("dispconv", 2)](x))
        
        x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
        if self.concatDepth:
            depth = F.interpolate(self.outputs[("disp", 2)], scale_factor=2, mode=self.upsample_mode)
            x = torch.cat((x, input_features[0], depth), 1)
        else:
            x = torch.cat((x, input_features[0]), 1)
        x = self.convs["conv4"](x)
        if self.doubleConv > 0:
            x = self.convs["doubleConv4"](x)
        if self.moreFeature:
            x = torch.cat((x, input_features[0]), 1)
            x = self.convs["conv45"](x)
        self.outputs[("disp", 1)] = self.sigmoid(self.convs[("dispconv", 1)](x))
        
        x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
        if self.concatDepth:
            depth = F.interpolate(self.outputs[("disp", 1)], scale_factor=2, mode=self.upsample_mode)
            x = torch.cat((x, depth), 1)
        x = self.convs["conv5"](x)
        if self.doubleConv > 0:
            x = self.convs["doubleConv5"](x)
        
        if self.moreConv == 1:
            x = self.convs["conv6"](x)
        elif self.moreConv == 2:
            x = self.convs["conv6"](x)
            x = self.convs["conv7"](x)
        self.outputs[("disp", 0)] = self.sigmoid(self.convs[("dispconv", 0)](x))
            

        return self.outputs
        

