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
from .auto_prompt_encoder import AutoPromptEncoder
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
        self.pos_pt_number = 5
        self.neg_pt_num = 5

        self.auto_prompt_encoder = AutoPromptEncoder()

        for param in self.prompt_encoder.parameters():
          param.requires_grad = False
        for param in self.mask_decoder.parameters():
          param.requires_grad = False

        # for param in self.image_encoder.parameters():
        #   param.requires_grad = False
        for n, value in self.image_encoder.named_parameters():
          if "pos_embed" not in n and "2.attn.rel_pos" not in n and "5.attn.rel_pos" not in n and "8.attn.rel_pos" not in n and "11.attn.rel_pos" not in n and "samct" not in n:
            value.requires_grad = False
  

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self, 
        imgs: torch.Tensor,
        pt: Tuple[torch.Tensor, torch.Tensor] = None,  # [b n 2, b n]
        bbox: torch.Tensor=None,
    ) -> torch.Tensor:
        imge, imge_full, x256, x128, x64, x32, x16 = self.image_encoder(imgs)

        # In the outputs here, only se is related to manual prompts
        se, de, not_a_point_embed, point_embeddings = self.prompt_encoder(            # se b N 256, de b 256 32 32
                      points=pt,
                      boxes=bbox,
                      masks=None,
                  )
        
        pos_pt_embed, neg_pt_embed, box_embed, class_score, feature = self.auto_prompt_encoder(x256, x128, x64, x32, x16, imge, not_a_point_embed, point_embeddings) # only for the auto-prompt
        auto_se = torch.cat([pos_pt_embed, neg_pt_embed, box_embed], dim=1)

        # The input variables for the following processes are automatically generated
        low_res_masks, _ = self.mask_decoder(
                  image_embeddings=feature,
                  image_full_embedding=imge_full,
                  image_pe=self.prompt_encoder.get_dense_pe(), 
                  sparse_prompt_embeddings=auto_se, # The prompt embedding here is an automatically generated embedding
                  dense_prompt_embeddings=de, # "de" is generated from masks, we provide None to masks, so this embedding can be achieved without any manual prompts
                  multimask_output=False,
                  )

        masks = low_res_masks
        outputs = {"low_res_logits": low_res_masks, "masks": masks, "class_score":class_score, "pos_pt_embed":pos_pt_embed, "neg_pt_embed":neg_pt_embed, "box_embed":box_embed, "pos_pt_se":se[:, 1:2, :], "neg_pt_se":se[:, 2:3, :], "box_se":se[:, 3:, :]}
        return outputs

    def inference(
        self, 
        imgs: torch.Tensor,
        pt: Tuple[torch.Tensor, torch.Tensor],  # [b n 2, b n]
        bbox: torch.Tensor=None,
    ) -> torch.Tensor:
        imge, imge_full, x256, x128, x64, x32, x16 = self.image_encoder(imgs)

        se, de, not_a_point_embed, point_embeddings = self.prompt_encoder(  # se [b N 256], de [b 256 32 32]
            points=None,
            boxes=None,
            masks=None,
        )

        pos_pt_embed, neg_pt_embed, box_embed, class_score, feature = self.auto_prompt_encoder(x256, x128, x64, x32, x16, imge, not_a_point_embed, point_embeddings) # only for the auto-prompt # only for the auto-prompt
        auto_se = torch.cat([pos_pt_embed, neg_pt_embed, box_embed], dim=1)

        low_res_masks, _ = self.mask_decoder( # low_res_mask b 1 128 128
                  image_embeddings=feature,
                  image_full_embedding=imge_full,
                  image_pe=self.prompt_encoder.get_dense_pe(), 
                  sparse_prompt_embeddings=auto_se,
                  dense_prompt_embeddings=de, 
                  multimask_output=False,
                  )

        masks = low_res_masks
        outputs = {"low_res_logits": low_res_masks, "masks": masks, "pos_pt_embed":pos_pt_embed, "neg_pt_embed":neg_pt_embed, "box_pt_embed":box_embed, "pos_pt_se":se[:, :self.pos_pt_number, :], "neg_pt_se":se[:, self.pos_pt_number:self.pos_pt_number+self.neg_pt_num, :], "box_se":se[:, self.pos_pt_number+self.neg_pt_num:, :]}
        return outputs