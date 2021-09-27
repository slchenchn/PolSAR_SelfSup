'''
Author: Shuailin Chen
Created Date: 2021-09-14
Last Modified: 2021-09-25
	content: 
'''

import os.path as osp

import mmcv
import numpy as np
from PIL import Image
import re
from mmcv.utils import print_log

from openselfsup.utils import get_root_logger
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SN6SARProDataset(CustomDataset):
    """SpaceNet6 SAR-Pro dataset.
    """

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.tif',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)
        
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_name = osp.splitext(img_name)[0]
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name.replace('SAR-Intensity', 'PS-RGB') \
                                    + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix) \
                                    .replace('SAR-Intensity', 'PS-RGB')
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        # TODO: it seems that get_root_logger() doesn't work
        print_log(f'Loaded {len(img_infos)} images', logger='openselfsup')
        return img_infos