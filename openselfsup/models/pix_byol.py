'''
Author: Shuailin Chen
Created Date: 2021-09-10
Last Modified: 2021-09-23
	content: 
'''
import os.path as osp
import torch
from torch import nn, Tensor
import numpy as np
from PIL import Image
import mylib.labelme_utils as lu
import torchvision.transforms.functional as _transF

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

    def _view_img_mask_batch(self, 
                            img:Tensor,
                            mask:Tensor,
                            save_dir='tmp',
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],):
        ''' view a batch of images and masks '''

        assert img.dim()==5, f'expect #dim of img to be 5, got {img.dim()}'
        assert mask.dim()==4, f'expect #dim of mask to be 4, got {mask.dim()}'
        assert img.shape[1]==mask.shape[1]==2

        mean = Tensor(mean)
        std = Tensor(std)
        unnorm_mean = -mean / std
        unnorm_std = 1.0 / std

        B, V, C, H, W = img.shape
        for ii in range(B):
            for jj in range(V):
                img_path = osp.join(save_dir, f'{ii}_{jj}_img.png')
                mask_path = osp.join(save_dir, f'{ii}_{jj}_mask.png')
                im = img[ii, jj, ...]
                unnormed_im = _transF.normalize(im, unnorm_mean,
                                                unnorm_std, inplace=False)
                unnormed_im = unnormed_im.cpu().numpy().transpose(1, 2, 0)
                unnormed_im = (unnormed_im*255).astype(np.uint8)
                mk = mask[ii, jj, ...].cpu().numpy()
                Image.fromarray(unnormed_im).save(img_path)
                lu.lblsave(mask_path, mk)

    def forward_train(self, img, mask, **kargs):
        # self._view_img_mask_batch(img, mask)
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