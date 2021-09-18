'''
Author: Shuailin Chen
Created Date: 2021-09-10
Last Modified: 2021-09-18
	content: 
'''
from ..registry import DATASOURCES
from .image_list import ImageList


@DATASOURCES.register_module()
class SARCD(ImageList):

    def __init__(self, root, list_file, memcached, mclient_path, return_label=False, *args, **kwargs):
        super().__init__(
            root, list_file, memcached, mclient_path, return_label)
