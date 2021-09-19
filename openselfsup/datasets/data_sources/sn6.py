'''
Author: Shuailin Chen
Created Date: 2021-09-19
Last Modified: 2021-09-19
	content: 
'''

import os
import os.path as osp
import mylib.file_utils as fu
from PIL import Image

from openselfsup.utils import print_log
from ..registry import DATASOURCES
from .utils import McLoader
from .image_list import ImageList


@DATASOURCES.register_module(ImageList)
class SpaceNet6():
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
                mclient_path=None, 
                return_label=True):

        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.data_root = root
        self.list_file = list_file

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)

        assert osp.isdir(root), f'wrong data root of {root}'
        assert osp.isdir(img_dir), f'wrong image dir of {img_dir}'
        assert osp.isdir(ann_dir), f'wrong annotation dir of {ann_dir}'

        self.fns = []
        if isinstance(list_file, str):
            list_file = [list_file]
        for fi in list_file:
            self.fns.extend(fu.read_file_as_list(fi))

        self.memcached = memcached
        self.mclient_path = mclient_path
        self.initialized = False
        self.return_label = return_label

        self._init_memcached()
        print_log(f'totally {len(self.fns)} training sampls')

    def get_sample(self, idx):
        if self.memcached:
            raise NotImplementedError
        else:
            img = Image.open(osp.join(self.img_dir, self.fns[idx]))

        img = img.convert('RGB')
        
        if self.return_label:
            label = Image.open(osp.join(self.ann_dir, self.fns[idx]))
            return img, label
        else:
            return img
        