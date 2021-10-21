'''
Author: Shuailin Chen
Created Date: 2021-09-19
Last Modified: 2021-10-21
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
from typing import Tuple, List, Optional
import math

from openselfsup.utils import build_from_cfg
from ..registry import PIPELINES


def split_img_mask(img_mask):
    ''' Split image and mask, where image has 3 channels, mask has one channels

    Returns:
        PIL Image objects or Tensor
    '''
    if isinstance(img_mask, Image.Image):
        img_mask = np.asarray(img_mask)

    if isinstance(img_mask, ndarray):
        ''' shape of [H, W, C] '''
        assert img_mask.ndim==3, f'img_mask variable should has ndim=3, got {img_mask.ndim}'
        assert img_mask.shape[-1]==4, f'img_mask variable should has #channels=4, got {img_mask.shape[-1]} '
        
        img = img_mask[..., :3]
        mask = img_mask[..., -1]
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
    elif isinstance(img_mask, Tensor):
        ''' shape of [C, H, W] '''
        assert img_mask.ndim==3, f'img_mask variable should has ndim=3, got {img_mask.ndim}'
        assert img_mask.shape[0]==4, f'img_mask variable should has #channels=4, got {img_mask.shape[0]} '
        
        img = img_mask[:3, ...]
        mask = img_mask[3, ...]
    else:
        raise NotImplementedError

    return img, mask


def merge_img_mask(img, mask):
    ''' Merge image and mask, where image has 3 channels, mask has one channels

    Returns:
        PIL Image object or Tensor
    '''
    if isinstance(img, Image.Image):
        img = np.asarray(img)
        mask = np.asarray(mask)
        if mask.ndim==2:
            mask = mask[..., None]

        assert mask.ndim==img.ndim==3, f'img and mask should has ndim=3, got img.ndim={img.ndim}, mask.ndim={mask.ndim}'
        assert mask.shape[2]==1 and img.shape[2]==3, f'img and mask should has #channels=3 and 1 got mask.#channels={mask.shape[2]}, img.#channels={img.shape[2]} '
        
        img_mask = np.concatenate((img, mask), axis=-1)
        img_mask = Image.fromarray(img_mask)
    elif isinstance(img, Tensor):
        if mask.ndim==2:
            mask = mask[None, ...]

        assert mask.ndim==img.ndim==3, f'img and mask should has ndim=3, got img.ndim={img.ndim}, mask.ndim={mask.ndim}'
        assert mask.shape[0]==1 and img.shape[0]==3, f'img and mask should has #channels=3 and 1 got mask.#channels={mask.shape[0]}, img.#channels={img.shape[0]} '

        img_mask = torch.cat((img, mask), dim=0)
    else:
        raise NotImplementedError
        
    return img_mask


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


@PIPELINES.register_module()
class IMRandomResizedCrop(_transforms.RandomResizedCrop):
    '''Crop a random portion of image and mask, and resize it to a given size.

    Args:
        min_valid_ratio (float): minimum acceptable ratio of valid region.
            It will iterata until ratio of valid region greater than this.
    NOTE: interpolation methods of image is bilinear, of mask is nearest
    '''

    def __init__(self, *args, min_valid_ratio=0.25, **kargs):
        super().__init__(*args, **kargs)
        self.min_valid_ratio = min_valid_ratio

    @staticmethod
    def get_params(mask: Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            mask (PIL Image or Tensor): Input mask image
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = _transF._get_image_size(mask)
        area = height * width

        bi_mask = torch.from_numpy(np.asarray(mask)) > 0
        row, col = torch.nonzero(bi_mask, as_tuple=True)
        h_min = row[0]
        h_max = row[-1]
        w_min = col[0]
        w_max = col[-1]

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                # i = torch.randint(0, height - h + 1, size=(1,)).item()
                # j = torch.randint(0, width - w + 1, size=(1,)).item()
                i = torch.randint(h_min, h_max - h + 1, size=(1,)).item()
                j = torch.randint(w_min, w_max - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img_mask):
        assert isinstance(img_mask, Image.Image)
        img, mask = split_img_mask(img_mask)
        
        for _ in range(10):
            i, j, h, w = self.get_params(mask, self.scale, self.ratio)
            new_mask = _transF.resized_crop(mask, i, j, h, w, self.size,
                            interpolation=_transF.InterpolationMode.NEAREST)
            if (np.asarray(new_mask)>0).sum() > self.min_valid_ratio * np.prod(self.size):
                new_img = _transF.resized_crop(img, i, j, h, w, self.size,
                                            interpolation=self.interpolation)
                img_mask = merge_img_mask(new_img, new_mask)
                return img_mask


@PIPELINES.register_module()
class IMToTensor(_transforms.ToTensor):
    '''Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This transform does not support torchscript.
    '''

    def __call__(self, img_mask):
        img, mask = split_img_mask(img_mask)
        new_img = super().__call__(img)

        new_mask = np.expand_dims(np.asarray(mask), 0)
        new_mask = torch.from_numpy(new_mask).contiguous()
        img_mask = merge_img_mask(new_img, new_mask)
        return img_mask
