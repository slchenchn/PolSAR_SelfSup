'''
Author: Shuailin Chen
Created Date: 2021-09-19
Last Modified: 2021-11-19
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
import mmcv

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
        PIL Image object or Tensor or ndarray
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
    elif isinstance(img, ndarray):
        if mask.ndim==2:
            mask = mask[..., None]

        assert mask.ndim==img.ndim==3, f'img and mask should has ndim=3, got img.ndim={img.ndim}, mask.ndim={mask.ndim}'
        assert mask.shape[2]==1 and img.shape[2]==3, f'img and mask should has #channels=3 and 1 got mask.#channels={mask.shape[0]}, img.#channels={img.shape[0]} '

        img_mask = np.concatenate((img, mask), axis=-1)
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
    NOTE: interpolation methods of image is bilinear, of mask is nearest.
    It actually the v1 version: torchvision version
    '''

    def __init__(self, *args, min_valid_ratio=0.25, **kargs):
        super().__init__(*args, **kargs)
        self.min_valid_ratio = min_valid_ratio

    def forward(self, img_mask):
        assert isinstance(img_mask, Image.Image)
        img, mask = split_img_mask(img_mask)
        
        # for _ in range(10):
        while True:
            i, j, h, w = self.get_params(mask, self.scale, self.ratio)
            new_mask = _transF.resized_crop(mask, i, j, h, w, self.size,
                            interpolation=_transF.InterpolationMode.NEAREST)
            if (np.asarray(new_mask)>0).sum() > self.min_valid_ratio * np.prod(self.size):
                new_img = _transF.resized_crop(img, i, j, h, w, self.size,
                                            interpolation=self.interpolation)
                img_mask = merge_img_mask(new_img, new_mask)
                return img_mask


@PIPELINES.register_module()
class IMRandomCrop(object):
    """Random crop the image & mask. MMSeg implement version

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    def get_crop_bbox(self, mask):
        """Randomly get a crop bounding box, constrained by the valid mask
        
        Args:
            mask (ndarray): should be mask image, not original image
        """

        col, row = np.nonzero(mask>0)
        w_min = min(max(row.min() - int(self.crop_size[1]*0.5), 0), mask.shape[1]-self.crop_size[1])
        w_max = max(min(row.max() - int(self.crop_size[1]*0.5), mask.shape[1]-self.crop_size[1]), 0)
        h_min = min(max(col.min() - int(self.crop_size[0]*0.5), 0), mask.shape[0]-self.crop_size[0])
        h_max = max(min(col.max() - int(self.crop_size[0]*0.5), mask.shape[0]-self.crop_size[0]), 0)
        offset_h = np.random.randint(h_min, h_max+1)
        offset_w = np.random.randint(w_min, w_max+1)

        # margin_h = max(mask.shape[0] - self.crop_size[0], 0)
        # margin_w = max(mask.shape[1] - self.crop_size[1], 0)
        # offset_h = np.random.randint(0, margin_h + 1)
        # offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, img_mask):
        assert isinstance(img_mask, Image.Image)
        img, mask = split_img_mask(img_mask)
        img = np.asarray(img)
        mask = np.asarray(mask)

        crop_bbox = self.get_crop_bbox(mask)
        # if self.cat_max_ratio < 1.:
        #     # Repeat 10 times
        #     for _ in range(10):
        #         seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
        #         labels, cnt = np.unique(seg_temp, return_counts=True)
        #         cnt = cnt[labels != self.ignore_index]
        #         if len(cnt) > 1 and np.max(cnt) / np.sum(
        #                 cnt) < self.cat_max_ratio:
        #             break
        #         crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        mask = self.crop(mask, crop_bbox)
        img_mask = merge_img_mask(img, mask)
        return Image.fromarray(img_mask)


@PIPELINES.register_module()
class IMResize(object):
    """Resize images & seg. mmseg implementation

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can be None, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
            Default:None.
        multiscale_mode (str): Either "range" or "value".
            Default: 'range'
        ratio_range (tuple[float]): (min_ratio, max_ratio).
            Default: None
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: True
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], results['scale'], interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(
                    results[key], results['scale'], interpolation='nearest')
            results[key] = gt_seg

    def __call__(self, img_mask):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        # construt a result dict in favor of mmseg original implememtation
        img, mask = split_img_mask(img_mask)
        results = dict(img=np.asarray(img), 
                    gt_semantic_seg=np.asarray(mask),
                    seg_fields=['gt_semantic_seg'])
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        
        img_mask = merge_img_mask(results['img'], results['gt_semantic_seg'])
        return Image.fromarray(img_mask)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str


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
