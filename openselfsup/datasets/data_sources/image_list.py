'''
Author: Shuailin Chen
Created Date: 2021-09-14
Last Modified: 2021-09-23
	content: 
'''
import os
from PIL import Image

from openselfsup.utils import print_log, get_root_logger
from ..registry import DATASOURCES
from .utils import McLoader


@DATASOURCES.register_module()
class ImageList(object):
    ''' 
    Args:
        list_file (list|str): train split files
    '''

    def __init__(self, root, list_file, memcached=False, mclient_path=None, return_label=True):
        if isinstance(list_file, str):
            list_file = [list_file]
        train_files = []
        for fi in list_file:
            with open(fi, 'r') as f:
                lines = f.readlines()
            self.has_labels = len(lines[0].split()) == 2
            self.return_label = return_label
            if self.has_labels:
                self.fns, self.labels = zip(*[l.strip().split() for l in lines])
                self.labels = [int(l) for l in self.labels]
            else:
                assert self.return_label is False, f'return_label is True, but labels are not exist in split files'
                self.fns = [l.strip() for l in lines]
            self.fns = [os.path.join(root, fn) for fn in self.fns]
            train_files.extend(self.fns)

        self.fns = train_files
        self.memcached = memcached
        self.mclient_path = mclient_path
        self.initialized = False

        print_log(f'totally {len(self.fns)} training sampls',
                logger=get_root_logger())

    def _init_memcached(self):
        if not self.initialized:
            assert self.mclient_path is not None
            self.mc_loader = McLoader(self.mclient_path)
            self.initialized = True

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        if self.memcached:
            self._init_memcached()
        if self.memcached:
            img = self.mc_loader(self.fns[idx])
        else:
            img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        if self.has_labels and self.return_label:
            target = self.labels[idx]
            return img, target
        else:
            return img
