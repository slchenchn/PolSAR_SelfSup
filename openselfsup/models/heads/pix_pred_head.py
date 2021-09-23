'''
Author: Shuailin Chen
Created Date: 2021-09-18
Last Modified: 2021-09-22
	content: 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..registry import HEADS
from ..import builder
from .latent_pred_head import LatentPredictHead


@HEADS.register_module()
class PixPredHead(LatentPredictHead):
    ''' Head of pixels-level BYOL
    '''

    def forward(self, input, target, mask):
        """
        Args:
            input (Tensor): NxHxWxC input features.
            target (Tensor): NxHxWxC target features.
            mask (Tensor): NxHxW mask image.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = self.predictor([input])[0]
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)

        # n_segments = mask.max(dim=(1,2,3))
        n_segments = mask.max()
        # NOTE: Advanced indexing always returns a copy of the data
        for ii in range(1, n_segments+1):
            segment_feat = target[n_segments==ii]
        
        loss = 0
        if self.size_average:
            loss /= input.size(0)
        return dict(loss=loss)
        