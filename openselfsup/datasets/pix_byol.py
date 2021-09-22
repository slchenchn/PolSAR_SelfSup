'''
Author: Shuailin Chen
Created Date: 2021-09-19
Last Modified: 2021-09-19
	content: 
'''
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import numpy as np
from PIL import Image

from openselfsup.utils import build_from_cfg
from .registry import DATASETS, PIPELINES
from .builder import build_datasource
from .byol import BYOLDataset


@DATASETS.register_module()
class PixBYOLDataset(BYOLDataset):
    """Dataset for BYOL.
    """

    def __init__(self, return_label=True, **kargs):
        super().__init__(return_label, **kargs)

    @staticmethod
    def cat_pil_images(self, images, axis):
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

        img_cat = torch.cat((img_label1.unsqueeze(0), img_label2.unsqueeze(0)), dim=0)
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented
