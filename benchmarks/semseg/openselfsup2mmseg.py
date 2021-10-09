'''
Author: Shuailin Chen
Created Date: 2021-09-27
Last Modified: 2021-10-03
	content: 
'''

import os
import os.path as osp
import argparse
import torch
import mmcv
from collections import OrderedDict
import warnings
from colorama import Fore, Style



def convert_openselfsup(ckpt: OrderedDict):
    ''' Convert model parameters of OpenSelfSup format into MMSeg, only support BYOL now
    
    Args:
        ckpt (OrderedDict): model weights
    '''

    new_ckpt = OrderedDict()
    converted = False
    for ii, (k, v) in enumerate(ckpt.items()):
        if k.startswith('backbone'):
            converted = True
            new_key = k.replace('backbone.', '')
            print(f'select {k} as {new_key}')
            # if 'num_batches_tracked' not in k:
            #     print(new_key)
            new_ckpt[new_key] = v

    return new_ckpt, converted


def convert_whole_folder(folder, prefix):
    ''' Convert all checkpoint files in a folder 
    
    Args:
        prefix (str): prefix added to the result pth file
    '''

    assert osp.isdir(folder), f'{folder} must be a folder'
    for pth in os.listdir(folder):
        if (pth.endswith('.pth')) and ('mmseg' not in pth) \
            and ('latest' not in pth):
            ckpt = torch.load(osp.join(folder, pth), map_location='cpu')
            state_dict = ckpt['state_dict']
            ckpt['state_dict'], converted = convert_openselfsup(state_dict)
            
            if not converted:
                warnings.warn(f'{pth} has not been converted')
            
            dst_pth = osp.join(folder, prefix+pth)
            print(f'{Fore.GREEN}saving {dst_pth}{Fore.RESET}')
            torch.save(ckpt, dst_pth)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Convert keys in OpenSelfSup pretrained models to'
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    # parser.add_argument('dst', help='save path')
    args = parser.parse_args()
    # if args.dst is None:

    if osp.isdir(args.src):
        convert_whole_folder(args.src, prefix='mmseg_')
    else:
        checkpoint = torch.load(args.src, map_location='cpu')
        state_dict = checkpoint['state_dict']
        checkpoint['state_dict'] = convert_openselfsup(state_dict)
        mmcv.mkdir_or_exist(osp.dirname(args.dst))
        torch.save(checkpoint, args.dst)