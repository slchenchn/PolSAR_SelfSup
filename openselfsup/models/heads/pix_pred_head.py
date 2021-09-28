'''
Author: Shuailin Chen
Created Date: 2021-09-18
Last Modified: 2021-09-28
	content: 
'''

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from openselfsup.ops import resize
from ..registry import HEADS
from ..import builder
from .latent_pred_head import LatentPredictHead


@HEADS.register_module()
class PixPredHead(LatentPredictHead):
    ''' Head of pixels-level BYOL
    '''

    def forward(self,
                input:Tensor,
                target:Tensor,
                mask_input:Tensor,
                mask_target:Tensor):
        """
        Args:
            input (Tensor): NxHxWxC input features.
            target (Tensor): NxHxWxC target features.
            mask (Tensor): NxHxW mask image.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        pred = self.predictor([input])[0]
        pred_norm = F.normalize(pred, dim=1)
        target_norm = F.normalize(target, dim=1)
        pred_norm = pred_norm.permute(0, 2, 3, 1)
        target_norm = target_norm.permute(0, 2, 3, 1)

        mask_input = resize(mask_input.unsqueeze(1).type(torch.float32),
                    pred_norm.shape[2:]).squeeze().type(torch.int)
        mask_target = resize(mask_target.unsqueeze(1).type(torch.float32),
                    pred_norm.shape[2:]).squeeze().type(torch.int)
        

        n_segments = min(mask_input.max(), mask_target.max()).item()
        segment_norm = torch.empty_like(pred_norm)
        # segment_norm = torch.zeros_like(pred_norm)
        loss_mask = torch.zeros_like(mask_input)
        for ii in range(1, n_segments+1):
            # project to feature map of input
            # NOTE: Advanced indexing always returns a copy of the data
            target_idx = (mask_target==ii)
            segment_feat = target_norm[target_idx]
            input_idx = (mask_input==ii)
            segment_norm[input_idx] = segment_feat.mean(dim=0)
            loss_mask += input_idx
            
        loss = -2 * (pred_norm * segment_norm)[loss_mask].sum()
        if self.size_average:
            loss /= loss_mask.sum()
        return dict(loss=loss)
        