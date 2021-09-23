'''
Author: Shuailin Chen
Created Date: 2021-06-21
Last Modified: 2021-09-23
	content: 
'''

import collections
import os.path as osp
from mmcv.utils import build_from_cfg
from torchvision.transforms import Compose
import torchvision.transforms.functional as _transF
import mylib.labelme_utils as lu
import numpy as np

from ..registry import PIPELINES
from .img_label_transforms import split_img_mask


@PIPELINES.register_module()
class ComposeWithVisualization(Compose):
    """Compose multiple transforms images with saving intermedia results sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """
    
    def __init__(self, 
                *args,
                if_visualize=False,
                mean=None,
                std=None,
                save_dir='tmp',
                **kargs):
        super().__init__(*args, **kargs)
        self.if_visualize = if_visualize
        self.save_dir = save_dir
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """Call function to apply transforms sequentially with saving intermedia results.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                raise ValueError
            
            if self.if_visualize \
                and (type(t).__name__!='LoadImagesFromFile') \
                and (type(t).__name__!='DefaultFormatBundle') :

                print(type(t).__name__)
                img_path = osp.join(self.save_dir, type(t).__name__+'_img.png')
                gt_path = osp.join(self.save_dir, type(t).__name__+'_gt.png')
                
                img, mask = split_img_mask(data)
                lu.lblsave(gt_path, mask)

                if np.asarray(img).dtype == np.uint8:
                    ''' Not normalized '''
                    assert (self.mean is None) and (self.std is None), \
                        f'should not un-normalize here'
                    img.save(img_path)
                elif np.asarray(img).dtype == np.float32:
                    ''' Normalized '''
                    assert (self.mean is not None) and (self.std is not None),\
                        f'should un-normalize here'
                    unnorm_mean = -self.mean / self.std
                    unnorm_std = 1.0 / self.std
                    unnormed_img = _transF.normalize(img, unnorm_mean,
                                                unnorm_std, inplace=False)
                    unnormed_img.save(img_path)
                else:
                    raise NotImplementedError

        return data

