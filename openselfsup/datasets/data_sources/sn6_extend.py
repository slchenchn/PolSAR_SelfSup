'''
Author: Shuailin Chen
Created Date: 2021-09-19
Last Modified: 2021-10-21
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
class SpaceNet6Extend(ImageList):
    ''' Data source of SpacetNet6 and its extend version
    
    Args:
        list_file (str|list): list file(s) containing the training samples
    '''
    def __init__(self, 
                root, 
                img_dir, 
                ann_dir, 
                list_file, 
                memcached=False,
                ann_suffix='.png',
                mclient_path=None, 
                return_label=True):

        # check input
        assert osp.isdir(root), f'wrong data root of {root}'
        if not isinstance(img_dir, (tuple, list)):
            img_dir = [img_dir]
        if not isinstance(ann_dir, (tuple, list)):
            img_dir = [ann_dir]

        self.data_root = root
        self.list_file = list_file
        self.ann_suffix = ann_suffix

        # join paths if data_root is specified
        self.ann_dir = []
        self.img_dir = []
        for im, an in zip(img_dir, ann_dir):
            if root is not None:
                if not osp.isabs(im):
                    self.img_dir.append(osp.join(root, im))
                if not (self.ann_dir is None or osp.isabs(an)):
                    self.ann_dir.append(osp.join(root, an))


        self.fns = []
        self.anns = []
        if isinstance(list_file, str):
            list_file = [list_file]
        for fi, im, an in zip(list_file, self.img_dir, self.ann_dir):
            fns = fu.read_file_as_list(fi)
            fns = [osp.join(im, fn) for fn in fns]
            anns = [osp.join(an, osp.split(osp.splitext(fn)[0])[1]+self.ann_suffix) for fn in fns]
            # fns = [fn.split('.')[0] for fn in fns]
            self.fns.extend(fns)
            self.anns.extend(anns)

        self.memcached = memcached
        self.mclient_path = mclient_path
        self.initialized = False
        self.return_label = return_label

        if memcached:
            self._init_memcached()
        print_log(f'totally {len(self.fns)} training sampls',
                logger='openselfsup')

    # def get_length(self):
    #     lengths = [len(f) for f in self.fns]
    #     return sum(lengths)

    def get_sample(self, idx):
        if self.memcached:
            raise NotImplementedError
        else:
            img_path = self.fns[idx]
            img = Image.open(img_path)

        img = img.convert('RGB')
        
        if self.return_label:
            label_path = self.anns[idx]
            label = Image.open(label_path)
            return img, label
        else:
            return img
        