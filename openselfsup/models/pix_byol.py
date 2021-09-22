'''
Author: Shuailin Chen
Created Date: 2021-09-10
Last Modified: 2021-09-22
	content: 
'''

import torch
from torch import nn
import numpy as np
from PIL import Image

from openselfsup.utils import print_log
from . import builder
from .registry import MODELS
from .byol import BYOL


@MODELS.register_module()
class PixBYOL(BYOL):
    ''' BYOL on pixel level, contrasting pixel with mask embeddigns
    '''
    
    def __init__(self,
                backbone,
                neck=None,
                head=None,
                pretrained=None,
                base_momentum=0.996,
                **kwargs):

        super().__init__(backbone, neck, head, pretrained, base_momentum,
                        **kwargs)

    def forward_train(self, img, mask, **kargs):
        assert img.dim() == 5, f"Input must have 5 dims, got: {img.dim()}"
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()
        mask_v1 = mask[:, 0, ...].contiguous()
        mask_v2 = mask[:, 1, ...].contiguous()
        
        # compute query features
        proj_online_v1 = self.online_net(img_v1)[0]
        proj_online_v2 = self.online_net(img_v2)[0]
        
        with torch.no_grad():
            # QUERY: why need to clone
            proj_target_v1 = self.target_net(img_v1)[0].clone().detach()
            proj_target_v2 = self.target_net(img_v2)[0].clone().detach()

        # NOTE: mask should according to target features
        loss = self.head(proj_online_v1, proj_target_v2, mask_v2)['loss'] + \
               self.head(proj_online_v2, proj_target_v1, mask_v1)['loss']
        return dict(loss=loss)