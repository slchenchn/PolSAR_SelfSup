'''
Author: Shuailin Chen
Created Date: 2021-09-23
Last Modified: 2021-10-10
	content: 
'''

import torch
import torch.nn as nn
from packaging import version
from mmcv.cnn import kaiming_init, normal_init

from ..registry import NECKS
from ..utils import build_norm_layer
from .necks import _init_weights


@NECKS.register_module()
class NonLinear1x1ConvNeck(nn.Module):
    ''' Neck for Pixel-BYOL
    '''
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 dropout_ratio=0.0,
                 ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(in_channels, hid_channels, 1),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(hid_channels, out_channels, 1)
        )
    
    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1, \
                f"expect len of inputs feautes to be 1, Got: {len(x)}"
        x = x[0]
        x = self.mlp(x)
        return [x]