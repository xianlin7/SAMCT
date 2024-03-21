# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from doctest import OutputChecker
from this import d
from tkinter import X
from unittest import skip
from unittest.mock import patch
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type
from .common import LayerNorm2d, MLPBlock, Adapter, SpatialSelfattentionBlock, ChannelSelfattentionBlock, Down, Up, SingleDown, SingleUp
import math
from einops import rearrange
from .common import softmax_one

class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        if patch_size == 8:
            self.patch_embed = PatchEmbed0(
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed(
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.img_size//patch_size, self.img_size//patch_size, embed_dim) # torch.zeros(1, 1024//16, 1024//16, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)
        
        base_dim = 48
        self.samct_cnn_embed = CNNEmbed(in_chans=in_chans, embed_dim=base_dim) # new to sam
        self.samct_cnndown1 = Down(base_dim, 2*base_dim) # 96
        self.samct_cnn2trans1 = CNN2Trans(dimq=embed_dim, dimkv=2*base_dim, num_heads=8, qkv_bias=qkv_bias, cnn_patch_size=patch_size//2)
        self.samct_trans2cnn1 = Trans2CNN(dimcnn=2*base_dim, dimtrans=embed_dim, cnn_patch_size=patch_size//2)

        self.samct_cnndown2 = Down(2*base_dim, 4*base_dim) # 192
        self.samct_cnn2trans2 = CNN2Trans(dimq=embed_dim, dimkv=4*base_dim, num_heads=8, qkv_bias=qkv_bias, cnn_patch_size=patch_size//4)
        self.samct_trans2cnn2 = Trans2CNN(dimcnn=4*base_dim, dimtrans=embed_dim, cnn_patch_size=patch_size//4)

        self.samct_cnndown3 = Down(4*base_dim, 8*base_dim) # 384
        self.samct_cnn2trans3 = CNN2Trans(dimq=embed_dim, dimkv=8*base_dim, num_heads=8, qkv_bias=qkv_bias, cnn_patch_size=patch_size//8)
        self.samct_trans2cnn3 = Trans2CNN(dimcnn=8*base_dim, dimtrans=embed_dim, cnn_patch_size=patch_size//8)

        self.samct_cnndown4 = Down(8*base_dim, 16*base_dim) # 768
        self.samct_cnn2trans4 = CNN2Trans(dimq=embed_dim, dimkv=16*base_dim, num_heads=8, qkv_bias=qkv_bias, cnn_patch_size=patch_size//16)
        self.samct_trans2cnn4 = Trans2CNN(dimcnn=16*base_dim, dimtrans=embed_dim, cnn_patch_size=patch_size//16)

        self.samct_up1 = Up(16*base_dim, 8*base_dim, bilinear=False) # 32*32
        self.samct_up2 = Up(8*base_dim, 4*base_dim, bilinear=False) # 64*64
        self.samct_up3 = Up(4*base_dim, 2*base_dim, bilinear=False) # 128*128
        self.samct_up4 = Up(2*base_dim, base_dim, bilinear=False) # 256*256
        self.samct_neck = nn.Conv2d(base_dim, out_chans//8, kernel_size=1, bias=False,)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
        self.factor = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1) # (b 3 256 256)
        cnnx = self.samct_cnn_embed(x) # (b c 256 256)
        transx = self.patch_embed(x) # (b 32 32 c)
       
        if self.pos_embed is not None:
            transx = transx + self.pos_embed # (b 16 16 c)

        transx = self.blocks[0](transx)
        transx = self.blocks[1](transx)
        transxi = self.blocks[2](transx)
        cnnx1 = self.samct_cnndown1(cnnx) # (b c1 128 128)
        transx = self.samct_cnn2trans1(transxi, cnnx1) + transxi
        cnnx1 = self.samct_trans2cnn1(transxi, cnnx1) + cnnx1

        transx = self.blocks[3](transx)
        transx = self.blocks[4](transx)
        transxi = self.blocks[5](transx)
        cnnx2 = self.samct_cnndown2(cnnx1) # (b c2 64 64)
        transx = self.samct_cnn2trans2(transxi, cnnx2) + transxi
        cnnx2 = self.samct_trans2cnn2(transxi, cnnx2) + cnnx2

        transx = self.blocks[6](transx)
        transx = self.blocks[7](transx)
        transxi = self.blocks[8](transx)
        cnnx3 = self.samct_cnndown3(cnnx2) # (b c3 32 32)
        transx = self.samct_cnn2trans3(transxi, cnnx3) + transxi
        cnnx3 = self.samct_trans2cnn3(transxi, cnnx3) + cnnx3

        transx = self.blocks[9](transx)
        transx = self.blocks[10](transx)
        transxi = self.blocks[11](transx)
        cnnx4 = self.samct_cnndown4(cnnx3) # (b c4 16 16)
        transx = self.samct_cnn2trans4(transxi, cnnx4) + transxi
        cnnx4 = self.samct_trans2cnn4(transxi, cnnx4) + cnnx4

        transx = transx.permute(0, 3, 1, 2)

        x = cnnx4
        x = self.samct_up1(x, cnnx3) 
        x = self.samct_up2(x, cnnx2)
        x = self.samct_up3(x, cnnx1)
        x = self.samct_up4(x, cnnx)
        x = self.samct_neck(x)

        transx = self.neck(transx)
        
        return transx, x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

        self.samct_MLP_Adapter = Adapter(dim, skip_connect=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        xn = self.norm2(x)
        x = x + self.mlp(xn)
        x = x + 0.5 * self.samct_MLP_Adapter(xn)

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv0 = self.qkv(x)
        qkv = qkv0.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = softmax_one(attn, dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x

class qkvAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q= nn.Linear(dim, dim, bias=qkv_bias)
        self.k= nn.Linear(dim, dim, bias=qkv_bias)
        self.v= nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v:torch.Tensor) -> torch.Tensor:
        B, H, W, _ = q.shape
        q = self.q(q).reshape(B, H * W, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B*self.num_heads, H*W, -1)
        k = self.k(k).reshape(B, H * W, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B*self.num_heads, H*W, -1)
        v = self.v(v).reshape(B, H * W, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B*self.num_heads, H*W, -1)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = softmax_one(attn, dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class CNNEmbed(nn.Module):
    """
    Image to CNN Embedding.
    """
    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.csb = ChannelSelfattentionBlock(in_chans, embed_dim)
        self.ssb = SpatialSelfattentionBlock(in_chans, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = self.csb(x)
        out = self.ssb(x, xc)
        return out


class PatchEmbed0(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=16, stride=(8, 8), padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, (256+8, 256+8), mode="bilinear", align_corners=False)
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


class CNN2Trans(nn.Module):
    """Achieved by a multi-head attention block with one query and mutiple keys."""

    def __init__(
        self,
        dimq: int,
        dimkv: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        cnn_patch_size: int = 2,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dimq // num_heads
        self.scale = head_dim**-0.5
        self.window_size = cnn_patch_size

        self.q= nn.Linear(dimq, dimq, bias=qkv_bias)
        self.k= nn.Linear(dimkv, dimq, bias=qkv_bias)
        self.v= nn.Linear(dimkv, dimq, bias=qkv_bias)
        self.proj = nn.Linear(dimq, dimq)
        self.combine = Adapter(D_features=dimq, skip_connect=False)

    def forward(self, x_trans: torch.Tensor, x_cnn: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x_trans.shape
        # each q corresponds to a window-set keys:
        q_in = rearrange(x_trans, 'b H W c -> (b H W) c').unsqueeze(dim=1)
        if self.window_size>0:
            kv_in = rearrange(x_cnn, 'b c (h m) (w n) -> (b h w) (m n) c', m=self.window_size, n=self.window_size)
        else:
            x_cnn = F.interpolate(x_cnn, (H, W), mode="bilinear", align_corners=False)
            kv_in = rearrange(x_cnn, 'b c (h m) (w n) -> (b h w) (m n) c', m=1, n=1)

        q = self.q(q_in)
        q = rearrange(q, 'B N (g d) -> B g N d', g=self.num_heads)
        k = self.k(kv_in)
        k = rearrange(k, 'B N (g d) -> B g N d', g=self.num_heads)
        v = self.v(kv_in)
        v = rearrange(v, 'B N (g d) -> B g N d', g=self.num_heads)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        #attn = attn.softmax(dim=-1)
        attn = softmax_one(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'B g N d -> B N (g d)')
        out = self.proj(out)
        out = rearrange(out, '(b H W) N c -> b H W (N c)', H=H, W=W)
        return self.combine(x_trans + out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out) # b 1 h w
        return self.sigmoid(out)

class Trans2CNN0(nn.Module):
    """Achieved by a spatial attention module"""

    def __init__(
        self,
        dimtrans: int,
        dimcnn: int,
        cnn_patch_size: int = 2,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
        """
        super().__init__()
        self.window_size = cnn_patch_size
        self.sa = SpatialAttention()
        self.fc = nn.Conv2d(dimcnn, dimtrans, kernel_size=1, bias=False)
        self.scale = dimtrans**0.5
        self.sigmoid = nn.Sigmoid()
        self.combine = Adapter(D_features=dimcnn, skip_connect=False)

    def forward(self, x_trans: torch.Tensor, x_cnn: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x_trans.shape
        trans = rearrange(x_trans, 'b H W c->b c H W')
        sa = self.sa(trans) # b 1 H W
        sa = rearrange(sa, 'b g H W -> (b H W) g') # (BHW 1)
        q = rearrange(trans, 'b c H W -> (b H W) c').unsqueeze(dim=1) # (bHW 1 c)
        k = rearrange(self.fc(x_cnn), 'b c (H m) (W n) -> (b H W) (m n) c', m=self.window_size, n=self.window_size) # (bHW kk c)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale # (bHW 1 kk)
        attn = softmax_one(attn, dim=-1) * self.window_size  # BHW 1 kk
        attn = attn * sa[:, :, None]
        attn = rearrange(attn, '(b H W) g (m n) -> b g (H m) (W n)', m=self.window_size, H=H, W=W)
        out = x_cnn * attn
        #out = self.combine(out)
        return out

class Trans2CNN(nn.Module):
    """Achieved by a spatial attention module"""

    def __init__(
        self,
        dimtrans: int,
        dimcnn: int,
        cnn_patch_size: int = 2,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
        """
        super().__init__()
        self.window_size = cnn_patch_size
        self.sa = SpatialAttention()
        self.fc = nn.Conv2d(dimcnn, dimtrans, kernel_size=1, bias=False)
        self.scale = dimtrans**0.5
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_trans: torch.Tensor, x_cnn: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x_trans.shape
        trans = rearrange(x_trans, 'b H W c->b c H W')
        sa = self.sa(trans) # b 1 H W
        sa = rearrange(sa, 'b g H W -> (b H W) g') # (BHW 1)
        q = rearrange(trans, 'b c H W -> (b H W) c').unsqueeze(dim=1) # (bHW 1 c)
        k = rearrange(self.fc(x_cnn), 'b c (H m) (W n) -> (b H W) (m n) c', m=self.window_size, n=self.window_size) # (bHW kk c)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale # (bHW 1 kk)
        attn = attn.softmax(dim=-1)
        attn = 0.5 + self.sigmoid(attn - 1/(self.window_size*self.window_size*1.0))  # BHW 1 kk
        attn = attn * sa[:, :, None]
        attn = rearrange(attn, '(b H W) g (m n) -> b g (H m) (W n)', m=self.window_size, H=H, W=W)
        out = x_cnn * attn
        return out

