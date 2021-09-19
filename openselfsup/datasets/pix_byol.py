'''
Author: Shuailin Chen
Created Date: 2021-09-19
Last Modified: 2021-09-19
	content: 
'''
import torch
from torch.utils.data import Dataset

from openselfsup.utils import build_from_cfg

from torchvision.transforms import Compose

from .registry import DATASETS, PIPELINES
from .builder import build_datasource
from .utils import to_numpy
from .byol import BYOLDataset


@DATASETS.register_module()
class PixBYOLDataset(BYOLDataset):
    """Dataset for BYOL.
    """

    def __init__(self, return_label=True, **kargs):
        super().__init__(return_label, **kargs)


    def __getitem__(self, idx):
        img, label = self.data_source.get_sample(idx)
        img_label = 
        img1 = self.pipeline1(img)
        img2 = self.pipeline2(img)
        if self.prefetch:
            img1 = torch.from_numpy(to_numpy(img1))
            img2 = torch.from_numpy(to_numpy(img2))

        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        return dict(img=img_cat)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented
