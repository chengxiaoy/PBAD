""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GroupNorm

from .encoder import build_eff_encoder


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        num_groups = 32
        if out_channels % num_groups != 0 or out_channels == num_groups:
            num_groups = out_channels // 2

        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3),
                      stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.blocks(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample=True, **_):
        super().__init__()
        self.up_sample = up_sample

        self.block = nn.Sequential(
            Conv3x3GNReLU(in_channels, out_channels),
            Conv3x3GNReLU(out_channels, out_channels),
        )

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x, skip = x
        else:
            skip = None

        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        return self.block(x)


class Decoder(nn.Module):
    def __init__(self,
                 num_classes,
                 encoder_channels,
                 dropout=0.2,
                 out_channels=[256, 128, 64, 32, 16],
                 **_):
        super().__init__()
        in_channels = encoder_channels
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.center = DecoderBlock(in_channels=in_channels[0], out_channels=in_channels[0],
                                   up_sample=False)

        self.layer1 = DecoderBlock(in_channels[0] + in_channels[1], out_channels[0])
        self.layer2 = DecoderBlock(in_channels[2] + out_channels[0], out_channels[1])
        self.layer3 = DecoderBlock(in_channels[3] + out_channels[1], out_channels[2])
        self.layer4 = DecoderBlock(in_channels[4] + out_channels[2], out_channels[3])
        self.layer5 = DecoderBlock(out_channels[3], out_channels[4])

        self.final = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(out_channels[-1], num_classes, kernel_size=1)
        )

    def forward(self, x, encodes):
        skips = encodes[1:]

        center = self.center(encodes[0])

        decodes = self.layer1([center, skips[0]])
        decodes = self.layer2([decodes, skips[1]])
        decodes = self.layer3([decodes, skips[2]])
        decodes = self.layer4([decodes, skips[3]])
        decodes = self.layer5([decodes, None])

        decodes4 = F.interpolate(decodes, x.size()[2:], mode='bilinear')
        outputs = self.final(decodes4)
        return outputs


class UNet_EFF(nn.Module):
    def __init__(self, encoder, num_classes, **kwargs):
        super().__init__()
        self.encoder = build_eff_encoder(encoder)
        self.decoder = Decoder(num_classes=num_classes, encoder_channels=self.encoder.channels, **kwargs)

    def forward(self, x):
        encodes = self.encoder(x)
        return self.decoder(x, encodes)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

