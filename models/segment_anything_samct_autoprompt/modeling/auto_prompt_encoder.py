# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from re import X
from tokenize import Double
import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d, softmax_one
from einops import rearrange


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

class vitAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, qx, kx):
        q = self.to_q(qx)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        kv = self.to_kv(kx).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn =  softmax_one(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm2in(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm1(x1), self.norm2(x2), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm2in(dim, vitAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x, vit):
        for attn, ff in self.layers:
            ax = attn(x, vit)
            x = ax + x
            x = ff(x) + x
        return x

class AutoPromptEncoder(nn.Module):
    def __init__(
        self,
        out_dim: int = 256,
        base_dim: int=48,
        num_heads: int = 8,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = out_dim
        self.base_dim = base_dim
        self.num_heads = num_heads
        self.scale = (out_dim//self.num_heads)**-0.5

        self.pos_protype = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.neg_protype = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.projection256 = nn.Sequential(
            nn.Conv2d(self.base_dim, self.embed_dim, kernel_size=1, bias=False),
            LayerNorm2d(self.embed_dim),
        )
        self.projection128 = nn.Sequential(
            nn.Conv2d(2*self.base_dim, self.embed_dim, kernel_size=1, bias=False),
            LayerNorm2d(self.embed_dim),
        )
        self.projection64 = nn.Sequential(
            nn.Conv2d(4*self.base_dim, self.embed_dim, kernel_size=1, bias=False),
            LayerNorm2d(self.embed_dim),
        )
        self.projection32 = nn.Sequential(
            nn.Conv2d(8*self.base_dim, self.embed_dim, kernel_size=1, bias=False),
            LayerNorm2d(self.embed_dim),
        )
        self.projection16 = nn.Sequential(
            nn.Conv2d(16*self.base_dim, self.embed_dim, kernel_size=1, bias=False),
            LayerNorm2d(self.embed_dim),
        )
        self.projectionViT = nn.Sequential(
            nn.Conv2d(out_dim, 2*self.embed_dim, kernel_size=1, bias=False),
            LayerNorm2d(2*self.embed_dim),
            nn.Conv2d(2*self.embed_dim, self.embed_dim, kernel_size=1, bias=False),
        )

        self.max_pool =  nn.AdaptiveMaxPool2d(1)
        self.avg_pool =  nn.AdaptiveAvgPool2d(1)

        self.pos_q= nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.pos_k= nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.pos_v= nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.pos_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.neg_q= nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.neg_k= nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.neg_v= nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.neg_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.to_pos_embedding = nn.Sequential(
             nn.Linear(self.embed_dim, self.embed_dim, bias=True),
             activation(),
             nn.Linear(self.embed_dim, out_dim, bias=True),
        )
        self.to_neg_embedding = nn.Sequential(
             nn.Linear(self.embed_dim, self.embed_dim, bias=True),
             activation(),
             nn.Linear(self.embed_dim, out_dim, bias=True),
        )
        self.to_box_lt_embedding = nn.Sequential(
             nn.Linear(self.embed_dim, self.embed_dim, bias=True),
             activation(),
             nn.Linear(self.embed_dim, out_dim, bias=True),
        )
        self.to_box_rb_embedding = nn.Sequential(
             nn.Linear(self.embed_dim, self.embed_dim, bias=True),
             activation(),
             nn.Linear(self.embed_dim, out_dim, bias=True),
        )

        self.to_class_score = nn.Linear(self.embed_dim, 2)
        normal_init(self.to_class_score, 0, 0.01)
        self.sig = nn.Softmax(dim=-1)

    def forward(self,
        feature256: torch.Tensor,
        feature128: torch.Tensor,
        feature64: torch.Tensor,
        feature32: torch.Tensor,
        feature16: torch.Tensor,
        featureViT: torch.Tensor,
        not_a_point_embed: torch.Tensor,
        point_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returning positive_point_embedding, negtive_point_embedding, and bbox_embedding

        Arguments:
          feature256: torch.Tensor with shape (B,basedim,256,256)
          feature128: torch.Tensor with shape (B,2*basedim,128,128)
          feature64: torch.Tensor with shape (B,4*basedim,64,64)
          feature32: torch.Tensor with shape (B,8*basedim,32,32)
          feature16: torch.Tensor with shape (B,16*basedim,32,32)
          featureViT: torch.Tensor with shape (B,embed_dim,16,16)

        Returns:
          torch.Tensor: sparse embeddings for the positive point, with shape Bx1x(embed_dim).
          torch.Tensor: sparse embeddings for the negtive point, with shape Bx1x(embed_dim).
          torch.Tensor: sparse embeddings for the box, with shape Bx1x(embed_dim).
        """
        bs = featureViT.shape[0]
        pos_tokens, neg_tokens = [], []
        feature = self.projection256(feature256) # b c h w
        pos_tokens.append(self.max_pool(feature))
        neg_tokens.append(self.avg_pool(feature))
        feature = self.projection128(feature128)
        pos_tokens.append(self.max_pool(feature))
        neg_tokens.append(self.avg_pool(feature))
        feature = self.projection64(feature64)
        pos_tokens.append(self.max_pool(feature))
        neg_tokens.append(self.avg_pool(feature))
        feature = self.projection32(feature32)
        pos_tokens.append(self.max_pool(feature))
        neg_tokens.append(self.avg_pool(feature))
        feature = self.projection16(feature16)
        pos_tokens.append(self.max_pool(feature))
        neg_tokens.append(self.avg_pool(feature))
        feature = self.projectionViT(featureViT)
        pos_tokens.append(self.max_pool(feature))
        neg_tokens.append(self.avg_pool(feature))

        pos_tokens = torch.stack(pos_tokens, dim=1) # shape of (B 6 c 1 1)
        pos_tokens = pos_tokens.reshape(bs, 6, -1)
        neg_tokens = torch.stack(neg_tokens, dim=1) # shape of (B 6 c 1 1)
        neg_tokens = neg_tokens.reshape(bs, 6, -1)
        pos_protype = self.pos_protype.repeat(bs, 1, 1)
        neg_protype = self.neg_protype.repeat(bs, 1, 1)

        pos_q = self.pos_q(pos_protype)
        pos_q = rearrange(pos_q, 'B N (g d) -> B g N d', g=self.num_heads)
        pos_k = self.pos_k(pos_tokens)
        pos_k = rearrange(pos_k, 'B N (g d) -> B g N d', g=self.num_heads)
        pos_v = self.pos_v(pos_tokens)
        pos_v = rearrange(pos_v, 'B N (g d) -> B g N d', g=self.num_heads)
        pos_attn = torch.matmul(pos_q, pos_k.transpose(-1, -2)) * self.scale # shape of (B g n n)
        #pos_attn = pos_attn.softmax(dim=-1)
        pos_attn = softmax_one(pos_attn, dim=-1)
        pos_out = torch.matmul(pos_attn, pos_v)
        pos_out = rearrange(pos_out, 'B g N d -> B N (g d)')
        pos_out = self.pos_proj(pos_out)

        neg_q = self.neg_q(neg_protype)
        neg_q = rearrange(neg_q, 'B N (g d) -> B g N d', g=self.num_heads)
        neg_k = self.neg_k(neg_tokens)
        neg_k = rearrange(neg_k, 'B N (g d) -> B g N d', g=self.num_heads)
        neg_v = self.neg_v(neg_tokens)
        neg_v = rearrange(neg_v, 'B N (g d) -> B g N d', g=self.num_heads)
        neg_attn = torch.matmul(neg_q, neg_k.transpose(-1, -2)) * self.scale # shape of (B g n n)
        #neg_attn = neg_attn.softmax(dim=-1)
        neg_attn = softmax_one(neg_attn, dim=-1)
        neg_out = torch.matmul(neg_attn, neg_v)
        neg_out = rearrange(neg_out, 'B g N d -> B N (g d)')
        neg_out = self.neg_proj(neg_out)

        class_score = self.to_class_score(pos_out) # b 1 2
        class_prob = self.sig(class_score)

        neg_pt_embedding = self.to_neg_embedding(neg_out)
        neg_pt_embedding =  neg_pt_embedding + point_embeddings[0].weight
        pos_pt_embedding = self.to_pos_embedding(pos_out) # b 1 d
        pos_pt_embedding = pos_pt_embedding + point_embeddings[1].weight

        pos_pt_embedding[class_prob[:, :, 0]>0.5, :] = not_a_point_embed.weight
    
        box_lt_embedding = self.to_box_lt_embedding(pos_out)
        box_lt_embedding = box_lt_embedding + point_embeddings[2].weight

        box_lt_embedding[class_prob[:, :, 0]>0.5, :] = not_a_point_embed.weight
        
        box_rb_embedding = self.to_box_rb_embedding(pos_out)
        box_rb_embedding =  box_rb_embedding + point_embeddings[3].weight

        box_rb_embedding[class_prob[:, :, 0]>0.5, :] = not_a_point_embed.weight

        #neg_pt_embedding = None
        box_embedding = torch.cat([box_lt_embedding, box_rb_embedding], dim=1)
        
        return pos_pt_embedding, neg_pt_embedding, box_embedding, class_score, feature


