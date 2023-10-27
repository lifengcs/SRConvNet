import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.m_block import *

def create_model(args):
    return SRNet(args)


class SRNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)
        self.scale = args.scale
        self.num_heads = args.num_heads
        self.num_kernels = args.num_kernels
        self.colors = args.colors
        self.dim = args.dim
        self.num_blocks = args.num_blocks

        self.to_feat = nn.Conv2d(self.colors, self.dim, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.Sequential(
            *[BasicBlock(self.dim, self.num_heads, self.num_kernels) for _ in range(self.num_blocks)]
        )

        if self.scale == 4:
            self.upsampling = nn.Sequential(
                nn.Conv2d(self.dim, self.dim * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(self.dim, self.dim * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.GELU()
            )
        else:
            self.upsampling = nn.Sequential(
                nn.Conv2d(self.dim, self.dim * self.scale * self.scale, 1, 1, 0),
                nn.PixelShuffle(self.scale),
                nn.GELU()
            )

        self.tail = nn.Conv2d(self.dim, self.colors, 3, 1, 1)

    def forward(self, x):
        base = x
        x = self.to_feat(x)
        x_init = x
        x = self.blocks(x) + x_init
        x = self.upsampling(x)
        x = self.tail(x)
        base = F.interpolate(base, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return x + base

    def load(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            name = name[name.index('.') + 1:]
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('upsampling') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('upsampling') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

