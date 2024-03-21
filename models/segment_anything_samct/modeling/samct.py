# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from turtle import shape
import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from einops import rearrange


class Samct(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        for param in self.prompt_encoder.parameters():
          param.requires_grad = False
        for param in self.mask_decoder.parameters():
          param.requires_grad = False
        # for param in self.image_encoder.parameters():
        #   param.requires_grad = False
        for n, value in self.image_encoder.named_parameters():
          if "pos_embed" not in n and "2.attn.rel_pos" not in n and "5.attn.rel_pos" not in n and "8.attn.rel_pos" not in n and "11.attn.rel_pos" not in n and "samct" not in n:
            value.requires_grad = False
        # for n, value in self.mask_decoder.named_parameters():
        #   if "iou_prediction_head" not in n:
        #     value.requires_grad = False
  

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self, 
        imgs: torch.Tensor,
        pt: Tuple[torch.Tensor, torch.Tensor],  # [b n 2, b n]
        bbox: torch.Tensor=None,
    ) -> torch.Tensor:
        imge, imge_full= self.image_encoder(imgs)
        
        if len(pt[0].shape) == 3:
          se, de = self.prompt_encoder(            # se [b (n+1) 256], de [b 256 32 32]
                        points=pt,
                        boxes=bbox,
                        masks=None,
                    )
          low_res_masks, _ = self.mask_decoder(
                    image_embeddings=imge,
                    image_full_embedding=imge_full,
                    image_pe=self.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False,
                    )
          masks = low_res_masks
          outputs = {"low_res_logits": low_res_masks, "masks": masks}
          return outputs
        else:
          low_res_masks, masks = [], []
          for i in range(pt[0].shape[1]):
            pti = (pt[0][:, i, :, :], pt[1][:, i, :])
            sei, dei = self.prompt_encoder(            # se b 2 256, de b 256 32 32
                        points=pti,
                        boxes=None,
                        masks=None,
                    )
            low_res_masksi, _ = self.mask_decoder(
                    image_embeddings=imge,
                    image_full_embedding=imge_full,
                    image_pe=self.prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=sei,
                    dense_prompt_embeddings=dei, 
                    multimask_output=False,
                    )
            masksi = low_res_masksi
            low_res_masks.append(low_res_masksi)
            masks.append(masksi)
          low_res_masks = torch.stack(low_res_masks, dim=1)
          masks = torch.stack(masks, dim=1) # b c 1 255 255
          masks = masks.reshape(masks.shape[0], -1, masks.shape[3], masks.shape[4])
          low_res_masks = low_res_masks.reshape(low_res_masks.shape[0], -1, low_res_masks.shape[3], low_res_masks.shape[4])
          outputs = {"low_res_logits": low_res_masks, "masks": masks}
          return outputs