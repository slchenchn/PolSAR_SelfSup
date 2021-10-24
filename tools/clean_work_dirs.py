'''
Author: Shuailin Chen
Created Date: 2021-08-08
Last Modified: 2021-10-24
	content: clear unnecessary work_dirs
'''

import os
import os.path as osp
from pathlib import Path
from glob import glob
import shutil
import argparse


def clean_no_run_dirs(path, bytes_threshold=600):
    ''' Clean work dirs for debug, where the model not actually run for a single iteration    

    Args:
        bytes_threshold (int): threshold of tensorboard file size, if the
            tensorboard file size smaller that this, it should be regard as uncessary work_dirs. Default: 50
    '''

    for timestamp in os.listdir(path):
        tf_file_name = glob(osp.join(path, timestamp, 'tf_logs', r'*'))

        if len(tf_file_name) > 1:
            raise NotImplementedError
        
        else:
            if (not tf_file_name) or osp.getsize(tf_file_name[0]) < bytes_threshold:
                ''' regard as invalid work dir '''
                print(f'remove dir {osp.join(path, timestamp)}')
                shutil.rmtree(osp.join(path, timestamp))


def clean_no_run_dirs_v2(path, bytes_threshold=600):
    ''' Clean work dirs for debug. The v1 version above can only clear timestamps under one work_dir, this v2 version clears all the work_dirs

    Args:
        bytes_threshold (int): threshold of tensorboard file size, if the
            tensorboard file size smaller that this, it should be regard as uncessary work_dirs. Default: 50
    '''

    work_dirs = glob(osp.join(path, r'**/tf_logs'), recursive=True)
    for dir in work_dirs:
        tf_file_name = glob(osp.join(dir, r'*'))

        if len(tf_file_name) > 1:
            raise NotImplementedError
        
        else:
            if (not tf_file_name) or osp.getsize(tf_file_name[0]) < bytes_threshold:
                ''' regard as invalid work dir '''
                rm_dir = osp.split(dir)[0]
                print(f'remove dir {rm_dir}, its tf file size is {osp.getsize(tf_file_name[0])}')
                shutil.rmtree(rm_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, 
            default=r'work_dirs/deeplabv3plus_512x512_800_mixbn_rs2_to_gf3')
    parser.add_argument('--threshold', type=int, 
            default=600)
    
    args = parser.parse_args()

    # clean_no_run_dirs(args.path, args.threshold)
    clean_no_run_dirs_v2(args.path, args.threshold)