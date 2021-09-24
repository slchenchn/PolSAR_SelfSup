'''
Author: Shuailin Chen
Created Date: 2021-09-18
Last Modified: 2021-09-24
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

    def forward(self, input:Tensor, target:Tensor, mask:Tensor):
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
        mask = resize(mask.unsqueeze(1).type(torch.float32),
                    pred_norm.shape[2:])
        mask = mask.squeeze().type(torch.int)
        pred_norm = pred_norm.permute(0, 2, 3, 1)
        target_norm = target_norm.permute(0, 2, 3, 1)

        # n_segments = mask.max(dim=(1,2,3))
        n_segments = mask.max().item()
        # segment_norm = torch.empty_like(pred_norm)
        segment_norm = torch.zeros_like(pred_norm)
        for ii in range(1, n_segments+1):
            # NOTE: Advanced indexing always returns a copy of the data
            pix_idx = (mask==ii)
            segment_feat = target_norm[pix_idx]
            segment_norm[mask==ii] = segment_feat.mean(dim=0)
            
        loss = -2 * (pred_norm * segment_norm).sum()
        if self.size_average:
            loss /= (mask>0).sum()
        return dict(loss=loss)
        