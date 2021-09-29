'''
Author: Shuailin Chen
Created Date: 2021-09-19
Last Modified: 2021-09-28
	content: 
'''

import os
import os.path as osp
import mylib.file_utils as fu
from PIL import Image

from openselfsup.utils import print_log, get_root_logger
from ..registry import DATASOURCES
from .utils import McLoader
from .image_list import ImageList


@DATASOURCES.register_module()
class SpaceNet6(ImageList):
    ''' Data source of SpacetNet6 
    
    Args:
        list_file (str|list): list file(s) containing the training samples
    '''
    def __init__(self, 
                root, 
                img_dir, 
                ann_dir, 
                list_file, 
                memcached=False,
                img_suffix='.tif',
                ann_suffix='.png',
                mclient_path=None, 
                return_label=True):

        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.data_root = root
        self.list_file = list_file
        self.img_suffix = img_suffix
        self.ann_suffix = ann_suffix

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)

        assert osp.isdir(root), f'wrong data root of {root}'
        assert osp.isdir(self.img_dir), f'wrong image dir of {self.img_dir}'
        assert osp.isdir(self.ann_dir), f'wrong annotation dir of {self.ann_dir}'

        self.fns = []
        if isinstance(list_file, str):
            list_file = [list_file]
        for fi in list_file:
            self.fns.extend(fu.read_file_as_list(fi))
        self.fns = [fn.split('.')[0] for fn in self.fns]

        self.memcached = memcached
        self.mclient_path = mclient_path
        self.initialized = False
        self.return_label = return_label

        if memcached:
            self._init_memcached()
        print_log(f'totally {len(self.fns)} training sampls',
                logger='openselfsup')

    def get_sample(self, idx):
        if self.memcached:
            raise NotImplementedError
        else:
            img_path = osp.join(self.img_dir, self.fns[idx]+self.img_suffix)
            img = Image.open(img_path)

        img = img.convert('RGB')
        
        if self.return_label:
            label_path = osp.join(self.ann_dir,
                                osp.split(self.fns[idx])[1]+self.ann_suffix)
            label = Image.open(label_path)
            return img, label
        else:
            return img
        