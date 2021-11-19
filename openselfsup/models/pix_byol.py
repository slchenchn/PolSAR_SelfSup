'''
Author: Shuailin Chen
Created Date: 2021-09-10
Last Modified: 2021-11-18
	content: 
'''
import os.path as osp
import torch
from torch import nn, Tensor
import numpy as np
from PIL import Image
import mylib.labelme_utils as lu
import torchvision.transforms.functional as _transF
from copy import deepcopy

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
class PixBYOLV5(PixBYOL):
    ''' PixBYOL v5, its mlp of projector and predictor should be 1x1 conv
    '''

    def forward_train(self, img, mask, **kargs):
        # self._view_img_mask_batch(img, mask)
        assert img.dim() == 5, f"Input must have 5 dims, got: {img.dim()}"
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()
        mask_v1 = mask[:, 0, ...].contiguous()
        mask_v2 = mask[:, 1, ...].contiguous()
        
        # compute query features
        proj_ol_v1 = self.online_net(img_v1)[0]
        proj_ol_v2 = self.online_net(img_v2)[0]
        
        with torch.no_grad():
            # backbone
            feat_tgt_v1 = self.backbone(img_v1)[0].permute(0, 2, 3, 1)
            feat_tgt_v2 = self.backbone(img_v2)[0].permute(0, 2, 3, 1)
            mask_v1 = resize(mask_v1.unsqueeze(1).float(),
                    feat_tgt_v1.shape[1:3]).squeeze().int()
            mask_v2 = resize(mask_v2.unsqueeze(1).float(),
                    feat_tgt_v1.shape[1:3]).squeeze().int()

            # extract segment feature
            segment_idx = np.intersect1d(mask_v1.cpu().numpy(), mask_v2.cpu().numpy())
            segment_feat_v1_v2 = torch.zeros_like(feat_tgt_v1)
            segment_feat_v2_v1 = torch.zeros_like(feat_tgt_v1)
            loss_mask_v1 = torch.zeros_like(mask_v1)
            loss_mask_v2 = torch.zeros_like(mask_v1)
            for ii in segment_idx:
                if ii==0:
                    continue

                # NOTE: Advanced indexing always returns a copy of the data
                idx_v1 = (mask_v1==ii)
                idx_v2 = (mask_v2==ii)
                segment_feat_v1_v2[idx_v2] = feat_tgt_v1[idx_v1].mean(dim=0)
                segment_feat_v2_v1[idx_v1] = feat_tgt_v2[idx_v2].mean(dim=0)
                loss_mask_v1 += idx_v1
                loss_mask_v2 += idx_v2

            # neck
            proj_tgt_v1_v2 = self.neck_tgt([segment_feat_v1_v2.permute(0, 3, 1, 2).contiguous()])[0]
            proj_tgt_v2_v1 = self.neck_tgt([segment_feat_v2_v1.permute(0, 3, 1, 2).contiguous()])[0]

        # NOTE: mask should according to target features
        loss = self.head(proj_ol_v1, proj_tgt_v2_v1, loss_mask_v1)['loss'] + self.head(proj_ol_v2, proj_tgt_v1_v2, loss_mask_v2)['loss']
        return dict(loss=loss, byol_momentum=torch.tensor([self.momentum], device=loss.device))


@MODELS.register_module()
class PPBYOLV5(BYOL):
    ''' PixBYOL + original BYOL
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

        # create neck and head of original BYOL
        aux_neck = deepcopy(head)
        aux_neck['type']='NonLinearNeckV2'
        aux_neck['with_avg_pool'] = True

        aux_head = deepcopy(head)
        aux_head['type'] = 'LatentPredictHead'
        aux_head['predictor']['with_avg_pool'] = False

        self.aux_neck_ol = builder.build_neck(aux_neck)
        self.aux_neck_tgt = builder.build_neck(aux_neck)
        self.aux_head_ol = builder.build_head(aux_head)

        for param in self.aux_neck_tgt.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the target network."""
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)
        
        for param_ol, param_tgt in zip(self.aux_neck_ol.parameters(),
                                       self.aux_neck_tgt.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

    def forward_train(self, img, mask, **kargs):
        # self._view_img_mask_batch(img, mask)
        assert img.dim() == 5, f"Input must have 5 dims, got: {img.dim()}"
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()
        mask_v1 = mask[:, 0, ...].contiguous()
        mask_v2 = mask[:, 1, ...].contiguous()
        
        # compute query features
        feat_ol_v1 = self.backbone(img_v1)
        feat_ol_v2 = self.backbone(img_v2)

        # original branch
        proj_ol_v1 = self.aux_neck_ol(feat_ol_v1)[0]
        proj_ol_v2 = self.aux_neck_ol(feat_ol_v2)[0]
        with torch.no_grad():
            feat_tgt_v1 = self.backbone_tgt(img_v1)
            feat_tgt_v2 = self.backbone_tgt(img_v2)
            proj_tgt_v1 = self.aux_neck_tgt(feat_tgt_v1)[0]
            proj_tgt_v2 = self.aux_neck_tgt(feat_tgt_v2)[0]

        ori_loss = self.head(proj_ol_v1, proj_tgt_v2)['loss'] \
            + self.head(proj_ol_v2, proj_tgt_v1)['loss']

        # PBYOL branch
        proj_ol_v1 = self.neck(feat_ol_v1)[0]
        proj_ol_v2 = self.neck(feat_ol_v2)[0]
        
        with torch.no_grad():
            # backbone
            feat_tgt_v1 = feat_tgt_v1.permute(0, 2, 3, 1)
            feat_tgt_v2 = feat_tgt_v2.permute(0, 2, 3, 1)
            mask_v1 = resize(mask_v1.unsqueeze(1).float(),
                    feat_tgt_v1.shape[1:3]).squeeze().int()
            mask_v2 = resize(mask_v2.unsqueeze(1).float(),
                    feat_tgt_v1.shape[1:3]).squeeze().int()

            # extract segment feature
            segment_idx = np.intersect1d(mask_v1.cpu().numpy(), mask_v2.cpu().numpy())
            segment_feat_v1_v2 = torch.zeros_like(feat_tgt_v1)
            segment_feat_v2_v1 = torch.zeros_like(feat_tgt_v1)
            loss_mask_v1 = torch.zeros_like(mask_v1)
            loss_mask_v2 = torch.zeros_like(mask_v1)
            for ii in segment_idx:
                if ii==0:
                    continue

                # NOTE: Advanced indexing always returns a copy of the data
                idx_v1 = (mask_v1==ii)
                idx_v2 = (mask_v2==ii)
                segment_feat_v1_v2[idx_v2] = feat_tgt_v1[idx_v1].mean(dim=0)
                segment_feat_v2_v1[idx_v1] = feat_tgt_v2[idx_v2].mean(dim=0)
                loss_mask_v1 += idx_v1
                loss_mask_v2 += idx_v2

            # neck
            proj_tgt_v1_v2 = self.neck_tgt([segment_feat_v1_v2.permute(0, 3, 1, 2).contiguous()])[0]
            proj_tgt_v2_v1 = self.neck_tgt([segment_feat_v2_v1.permute(0, 3, 1, 2).contiguous()])[0]

        # NOTE: mask should according to target features
        loss = self.head(proj_ol_v1, proj_tgt_v2_v1, loss_mask_v1)['loss'] + self.head(proj_ol_v2, proj_tgt_v1_v2, loss_mask_v2)['loss']
        return dict(ori_loss=ori_loss, ploss=loss, byol_momentum=torch.tensor([self.momentum], device=loss.device))

