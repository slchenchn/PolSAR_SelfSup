'''
Author: Shuailin Chen
Created Date: 2021-09-27
Last Modified: 2021-09-27
	content: 
'''

import os
import os.path as osp
import argparse
import torch
import mmcv
from collections import OrderedDict


def convert_openselfsup(ckpt: OrderedDict):
    ''' Convert model parameters of OpenSelfSup format into MMSeg, only support BYOL now
    
    Args:
        ckpt (OrderedDict): model weights
    '''


    new_ckpt = OrderedDict()

    for ii, (k, v) in enumerate(ckpt.items()):
        if k.startswith('backbone'):
            new_key = k.replace('backbone.', '')
            print(f'select {k} as {new_key}')
            # if 'num_batches_tracked' not in k:
            #     print(new_key)
            new_ckpt[new_key] = v

    return new_ckpt    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Convert keys in OpenSelfSup pretrained models to'
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    # if args.dst is None:

    checkpoint = torch.load(args.src, map_location='cpu')
    state_dict = checkpoint['state_dict']
    checkpoint['state_dict'] = convert_openselfsup(state_dict)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(checkpoint, args.dst)