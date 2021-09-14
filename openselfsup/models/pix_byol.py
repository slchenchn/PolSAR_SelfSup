'''
Author: Shuailin Chen
Created Date: 2021-09-10
Last Modified: 2021-09-10
	content: 
'''

import torch
from torch import nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS
from .byol import BYOL

@MODELS.register_module()
class PixBYOL(BYOL):
    ''' BYOL on pixel level, contrasting pixel with mask embeddigns
    '''