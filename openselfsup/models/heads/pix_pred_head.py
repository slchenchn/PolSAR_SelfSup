'''
Author: Shuailin Chen
Created Date: 2021-09-18
Last Modified: 2021-10-12
	content: 
'''

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
import numpy as np

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

        mask_input = resize(mask_input.unsqueeze(1).float(),
                    pred_norm.shape[1:3]).squeeze().int()
        mask_target = resize(mask_target.unsqueeze(1).float(),
                    pred_norm.shape[1:3]).squeeze().int()
        

        # n_segments = min(mask_input.max(), mask_target.max()).item()
        segment_idx = np.intersect1d(mask_input.cpu().numpy(), mask_target.cpu().numpy())
        segment_norm = torch.empty_like(pred_norm)
        # segment_norm = torch.zeros_like(pred_norm)
        loss_mask = torch.zeros_like(mask_input)
        for ii in segment_idx:
            if ii==0:
                continue

            # project to feature map of input
            # NOTE: Advanced indexing always returns a copy of the data
            target_idx = (mask_target==ii)
            segment_feat = target_norm[target_idx]
            input_idx = (mask_input==ii)
            segment_norm[input_idx] = segment_feat.mean(dim=0)
            # if torch.any(torch.isnan(segment_feat)):
            #     print('there are nan')
            # if torch.any(torch.isnan(segment_feat.mean(dim=0))):
            #     print(f'there are nan')
            loss_mask += input_idx
            
        loss = -2 * (pred_norm * segment_norm)[loss_mask.bool()].sum()
        
        if self.size_average:
            loss /= loss_mask.sum()
        return dict(loss=loss)
        