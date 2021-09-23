'''
Author: Shuailin Chen
Created Date: 2021-09-19
Last Modified: 2021-09-23
	content: 
'''

import os.path as osp
import cv2
import inspect
import numpy as np
from numpy import ndarray
from PIL import Image, ImageFilter
import torch
from torch import Tensor
from torchvision import transforms as _transforms
import torchvision.transforms.functional as _transF
import mylib.labelme_utils as lu

from openselfsup.utils import build_from_cfg
from ..registry import PIPELINES


def split_img_mask(img_mask):
    ''' Split image and mask, where image has 3 channels, mask has one channels

    Returns:
        PIL Image objects
    '''
    if isinstance(img_mask, Image.Image):
        img_mask = np.asarray(img_mask)

    assert img_mask.ndim==3, f'img_mask variable should has ndim=3, got {img_mask.ndim}'
    assert img_mask.shape[-1]==4, f'img_mask variable should has #channels=4,\
                                    got {img_mask.shape[-1]} '
    
    img = img_mask[..., :3]
    mask = img_mask[..., -1]
    return Image.fromarray(img), mask


def merge_img_mask(img, mask):
    ''' Merge image and mask, where image has 3 channels, mask has one channels

    Returns:
        PIL Image object
    '''
    if isinstance(img, Image):
        img = np.asarray(img)
    if isinstance(mask, Image):
        mask = np.asarray(mask)
    if mask.ndim==2:
        mask = mask[..., None]

    assert mask.ndim==3 and img.ndim==3, f'img and mask should has ndim=3, \
                    got img.ndim={img.ndim}, mask.ndim={mask.ndim}'
    assert mask.shape[-1]==1 and img.shape[-1]==3, f'img and mask should has \
        #channels=3 and 1 got mask.#channels={mask.shape[-1]}, img.#channels={img.shape[-1]} '
    
    img_mask = np.concatenate((img, mask), axis=-1)

    return Image.fromarray(img_mask)


@PIPELINES.register_module()
class IMRandomGrayscale(_transforms.RandomGrayscale):
    ''' Image-mask pair grayscale transformation '''

    def forward(self, img_mask):
        img, mask = split_img_mask(img_mask)
        new_img = super().forward(img)
        return merge_img_mask(new_img, mask)


@PIPELINES.register_module()
class RandomAppliedTransOnlyImg(object):
    """Randomly applied transformations to image only

    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float): Probability.
    """

    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.trans = _transforms.RandomApply(t, p=p)

    def __call__(self, img_mask):
        img, mask = split_img_mask(img_mask)
        new_img = self.trans(img)
        return merge_img_mask(new_img, mask)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class ViewImgLabels():
    ''' View the transformed images and labels
    '''

    def __init__(self, mean=None, std=None, save_dir='tmp') -> None:
        self.mean = mean
        self.std = std
        self.save_dir = save_dir

    def __call__(self, img_mask):
        img, mask = split_img_mask(img_mask)
        lu.lblsave(osp.join(self.save_dir, 'mask.png'), mask)
        if np.asarray(img).dtype == np.uint8:
            ''' Not normalized '''
            assert (self.mean is None) and (self.std is None), \
                f'should not un-normalize here'
            img.save(osp.join(self.save_dir, 'img.jpg'))
            # Image.fromarray(mask).save(osp.join(self.save_dir, 'mask.png'))
        elif np.asarray(img).dtype == np.float32:
            ''' Normalized '''
            assert (self.mean is not None) and (self.std is not None), \
                f'should un-normalize here'
            unnorm_mean = -self.mean / self.std
            unnorm_std = 1.0 / self.std
            # unnormed_img = _transforms.Normalize
            unnormed_img = _transF.normalize(img, unnorm_mean, unnorm_std,
                                            inplace=False)
            img.save(osp.join(self.save_dir, 'img.jpg'))
        else:
            raise NotImplementedError


@PIPELINES.register_module()
class IMNormalize(_transforms.Normalize):
    """ Normalize a tensor image only, this transform does not support PIL Image.

    NOTE: This transform acts out of place, i.e., it does not mutate the input tensor.
    """

    def forward(self, img_mask: Tensor) -> Tensor:
        img, mask = split_img_mask(img_mask)
        normed = super().forward(img)
        img_mask = merge_img_mask(normed, mask)
        return img_mask
