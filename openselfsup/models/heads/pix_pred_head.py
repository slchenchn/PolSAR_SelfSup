'''
Author: Shuailin Chen
Created Date: 2021-09-18
Last Modified: 2021-09-18
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

    def forward(self, input, target)