'''
Author: Shuailin Chen
Created Date: 2021-09-08
Last Modified: 2021-11-05
	content: 
'''
import torch
import torch.nn as nn

from openselfsup.utils import print_log, get_root_logger

from . import builder
from .registry import MODELS


@MODELS.register_module()
class BYOL(nn.Module):
    """BYOL.

    Implementation of "Bootstrap Your Own Latent: A New Approach to
    Self-Supervised Learning (https://arxiv.org/abs/2006.07733)".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        base_momentum (float): The base momentum coefficient for the target
            network. Default: 0.996.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 base_momentum=0.996,
                 **kwargs):
        super().__init__()
        self.online_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.target_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.backbone = self.online_net[0]
        self.neck = self.online_net[1]
        self.backbone_tgt = self.target_net[0]
        self.neck_tgt = self.target_net[1]
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.head = builder.build_head(head)
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

        self.online_net[0].init_weights(pretrained=pretrained) # backbone
        self.online_net[1].init_weights(init_linear='kaiming') # projection
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)
        # init the predictor in the head
        self.head.init_weights()

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the target network."""
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

    @torch.no_grad()
    def momentum_update(self):
        self._momentum_update()

    def forward_train(self, img, mask=None, **kwargs):
        """
        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C,
                H, W). Typically these should be mean centered and std scaled.
            mask (Tensor): superpixel mask on shape (N, 2, H, W)

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        assert img.dim() == 5, f"images must have 5 dims, got: {img.dim()}"
        assert mask is None or mask.dim() == 4, \
            f"masks must have 4 dims, got: {mask.dim()}"

        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()

        if mask is None:
            mask_v1 = None
            mask_v2 = None
        else:
            mask_v1 = mask[:, 0, ...].contiguous()
            mask_v2 = mask[:, 1, ...].contiguous()

        # compute query features
        proj_online_v1 = self.backbone(img_v1)
        proj_online_v1 = self.neck(proj_online_v1, mask_v1)[0]
        proj_online_v2 = self.backbone_tgt(img_v2)
        proj_online_v2 = self.neck(proj_online_v2, mask_v2)[0]

        with torch.no_grad():
            # QUERY: why need to clone
            proj_target_v1 = self.target_net(img_v1)[0].clone().detach()
            proj_target_v2 = self.target_net(img_v2)[0].clone().detach()

        loss = self.head(proj_online_v1, proj_target_v2)['loss'] \
            + self.head(proj_online_v2, proj_target_v1)['loss']
        return dict(loss=loss)

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
