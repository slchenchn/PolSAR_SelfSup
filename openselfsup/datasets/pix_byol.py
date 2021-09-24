'''
Author: Shuailin Chen
Created Date: 2021-09-19
Last Modified: 2021-09-23
	content: 
'''
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import numpy as np
from PIL import Image

from openselfsup.utils import build_from_cfg
from openselfsup.datasets.pipelines import ComposeWithVisualization
from .registry import DATASETS, PIPELINES
from .builder import build_datasource
from .byol import BYOLDataset
from openselfsup.datasets.pipelines.img_label_transforms import (
        split_img_mask, merge_img_mask)


@DATASETS.register_module()
class PixBYOLDataset(BYOLDataset):
    """ Dataset for PixBYOL
    """

    def __init__(self, data_source, pipeline1, pipeline2, 
                prefetch=False, 
                if_visualize=False):
        self.data_source = build_datasource(data_source)
        pipeline1 = [build_from_cfg(p, PIPELINES) for p in pipeline1]
        pipeline2 = [build_from_cfg(p, PIPELINES) for p in pipeline2]
        self.prefetch = prefetch

        self.pipeline1 = ComposeWithVisualization(pipeline1,
                                                if_visualize=if_visualize)
        self.pipeline2 = ComposeWithVisualization(pipeline2,
                                                if_visualize=if_visualize)

    @staticmethod
    def cat_pil_images(images, axis):
        arrays = [np.asarray(img) for img in images]
        arrays = [arr if arr.ndim==3 else arr[..., None] for arr in arrays]
        array = np.concatenate(arrays, axis=axis)
        return Image.fromarray(array)

    def __getitem__(self, idx):
        img, label = self.data_source.get_sample(idx)
        img_label = PixBYOLDataset.cat_pil_images((img, label), axis=-1)
        img_label1 = self.pipeline1(img_label)
        img_label2 = self.pipeline2(img_label)
        if self.prefetch:
            img_label1 = torch.from_numpy(np.asarray(img_label1))
            img_label2 = torch.from_numpy(np.asarray(img_label2))

        img1, mask1 = split_img_mask(img_label1)
        img2, mask2 = split_img_mask(img_label2)
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        mask_cat = torch.cat((mask1.unsqueeze(0), mask2.unsqueeze(0)), dim=0).type(torch.int)
        return dict(img=img_cat, mask=mask_cat)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented
