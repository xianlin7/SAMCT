# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True): #0.25
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        return x
        #return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# ----------------------- reference: https://github.com/CGPxy/AAU-net/tree/main -----------------------
class SpatialSelfattentionBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()
        )

        self.cbl2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()
        )

        self.sa = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(out_channels, 1, kernel_size=1, bias=False),
        )
        
        self.sigmoid = nn.Sigmoid()

        self.cb3 = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels)
        )

    def forward(self, x, xc):
        xs = self.cbl1(x)
        xs = self.cbl2(xs)

        data3 = xc + xs
        data3 = self.sa(data3)
        data3 = self.sigmoid(data3)
        a = data3.view(data3.shape[0], data3.shape[2], data3.shape[3])
        a1 = 1 - a
        y = xc * a[:, None, :, :]
        y1 = xs * a1 [:, None, :, :]
        data_a_a1 = torch.cat([y, y1], dim=1)
        output = self.cb3(data_a_a1)
        return output

class ChannelSelfattentionBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cbl1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()
        )

        self.cbl2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()
        )

        self.cbl3 = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, kernel_size=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()
        )

        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fcs = nn.Sequential(
            nn.Linear(out_channels*2, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        fd = self.cbl1(x)
        f5 = self.cbl2(x)
        data3 = torch.cat([fd, f5], dim=1)
        data3 = self.global_pooling(data3)
        data3 = data3.view(data3.shape[0], data3.shape[1])
        data3 = self.fcs(data3)
        data3 = self.sigmoid(data3)
        a = data3
        a1 = 1 - data3
        y = fd * a[:, :, None, None]
        y1 = f5 * a1 [:, :, None, None]
        data_a_a1 = torch.cat([y, y1], dim=1)
        output = self.cbl3(data_a_a1)
        return output

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SingleUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = SingleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = SingleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SingleDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()     #nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class SingleConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    #subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    #compute exponentials
    exp_x = torch.exp(x)
    #compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))
