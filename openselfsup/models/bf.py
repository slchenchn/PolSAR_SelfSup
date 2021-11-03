'''
Author: Shuailin Chen
Created Date: 2021-09-10
Last Modified: 2021-10-29
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
class BF(nn.Module):
    ''' BYOL wth filter auxiliary head
    '''
    
    def __init__(self,
                backbone,
                neck=None,
                head=None,
                auxiliary_head=None,
                pretrained=None,
                base_momentum=0.996,
                ):

        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)

        self.backbone_tgt = builder.build_backbone(backbone)
        self.neck_tgt = builder.build_neck(neck)
        for param in self.backbone_tgt.parameters():
            param.requires_grad = False
        for param in self.neck_tgt.parameters():
            param.requires_grad = False
        self.init_weights(pretrained=pretrained)

        self.base_momentum = base_momentum
        self.momentum = base_momentum

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {pretrained}',
                    logger='openselfsup')
        else:
            print_log(f'load model from None, traning from scratch',
                    logger='openselfsup')

        self.backbone.init_weights(pretrained=pretrained) # backbone
        self.neck.init_weights(init_linear='kaiming') # projection
        for param_ol, param_tgt in zip(self.backbone.parameters(),
                                       self.backbone_tgt.parameters()):
            param_tgt.data.copy_(param_ol.data)

        for param_ol, param_tgt in zip(self.neck.parameters(),
                                       self.neck_tgt.parameters()):
            param_tgt.data.copy_(param_ol.data)

        self.head.init_weights()

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
        laten_online_v1 = self.backbone(img_v1)
        proj_online_v1 = self.neck(laten_online_v1)[0]
        laten_online_v2 = self.backbone_tgt(img_v2)
        proj_online_v2 = self.neck_tgt(laten_online_v2)[0]

        
        
        with torch.no_grad():
            # QUERY: why need to clone
            proj_target_v1 = self.target_net(img_v1)[0].clone().detach()
            proj_target_v2 = self.target_net(img_v2)[0].clone().detach()

        # NOTE: mask should according to target features
        loss = self.head(proj_online_v1, proj_target_v2, mask_v1, mask_v2)['loss'] + self.head(proj_online_v2, proj_target_v1, mask_v2, mask_v1)['loss']
        return dict(loss=loss, byol_momentum=Tensor([self.momentum]))
    
    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))