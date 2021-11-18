'''
Author: Shuailin Chen
Created Date: 2021-09-10
Last Modified: 2021-11-17
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
from openselfsup.ops import resize
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
        loss = self.head(proj_online_v1, proj_target_v2, mask_v1, mask_v2)['loss'] + self.head(proj_online_v2, proj_target_v1, mask_v2, mask_v1)['loss']
        return dict(loss=loss, byol_momentum=torch.tensor([self.momentum], device=loss.device))


@MODELS.register_module()
class PixBYOLV5(BYOL):
    ''' PixBYOL v5
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
            feat_tgt_v1 = self.backbone(img_v1)[0]
            feat_tgt_v2 = self.backbone(img_v2)[0]

            B, C, H, W = img_v1.shape
            segment_pooled_feat_v1 = torch.zeros(B, 256, C, device=img_v1.device)
            segment_pooled_feat_v2 = segment_pooled_feat_v1.clone()

            for ii in range(256):
                if ii==0:
                    continue

                target_idx = (mask_v1==ii)
                segment_feat_v1 = feat_tgt_v1[target_idx]
                segment_pooled_feat_v1[ii] = segment_feat_v1.mean(dim=0)
                
                target_idx = (mask_v2==ii)
                segment_feat_v2 = feat_tgt_v2[target_idx]
                segment_pooled_feat_v2[ii] = segment_feat_v2.mean(dim=0)
            

        # NOTE: mask should according to target features
        loss = self.head(proj_online_v1, proj_target_v2, mask_v1, mask_v2)['loss'] + self.head(proj_online_v2, proj_target_v1, mask_v2, mask_v1)['loss']
        return dict(loss=loss, byol_momentum=torch.tensor([self.momentum], device=loss.device))
        
    @staticmethod
    def create_binary_mask(down_fac,
                        num_pixels,
                        mask,
                        max_mask_id=256,
                        downsample=(1, 32, 32, 1)):
        """Generates binary masks.

        From a mask of shape [batch_size, H,W] (values in range
        [0,max_mask_id], produces corresponding (downsampled) binary masks of
        shape [batch_size, max_mask_id, H*W/downsample] with biliear downsampling.
        Args:
            num_pixels: Number of points on the spatial grid
            masks: Felzenszwalb masks
            max_mask_id: # unique masks in Felzenszwalb segmentation
            downsample: rate at which masks must be downsampled.
        Returns:
            binary_mask: Binary mask with specification above
        """
        # convert to ont-hot
        B, H, W = mask.shape
        binary_mask = torch.zeros(B, max_mask_id, H/down_fac, W/down_fac,
                                device=mask.device)
        binary_mask.scatter_(-1, mask, 1)

        # downsample
        binary_mask = resize(binary_mask, scale_factor=down_fac, mode='bilinear')

        return binary_mask