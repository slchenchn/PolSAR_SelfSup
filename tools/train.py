'''
Author: Shuailin Chen
Created Date: 2021-09-09
Last Modified: 2021-10-26
	content: 
'''

from __future__ import division
import warnings
import argparse
import importlib
import os
import os.path as osp
from os import system
import time
from mylib.utils import wait_for_gpu

import mmcv
import torch
from mmcv import Config
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction
from mmcv.cnn.utils import revert_sync_batchnorm

from openselfsup import __version__
from openselfsup.apis import set_random_seed, train_model
from openselfsup.datasets import build_dataset
from openselfsup.models import build_model
from openselfsup.utils import collect_env, get_root_logger, traverse_replace
from benchmarks.semseg.openselfsup2mmseg import convert_whole_folder


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--work_dir',
        type=str,
        default=None,
        help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--pretrained', default=None, help='pretrained model file')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument('--port', type=int, default=29500,
        help='port only works when launcher=="slurm"')

    parser.add_argument('--required', type=float, default=None)
    parser.add_argument('--interval', type=float, default=10)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    wait_for_gpu(args.required, args.interval)

    cfg = Config.fromfile(args.config)
            
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    # remove functional key-value pairs
    uncessary_cfg_keys = ('deepcopy', 'copy')
    for k in uncessary_cfg_keys:
        cfg.pop(k, None)
        
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    # check memcached package exists
    if importlib.util.find_spec('mc') is None:
        traverse_replace(cfg, 'memcached', False)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        assert cfg.model.type not in \
            ['DeepCluster', 'MOCO', 'SimCLR', 'ODC', 'NPID'], \
            f"{cfg.model.type} does not support non-dist training."
    else:
        distributed = True
        if args.launcher == 'slurm':
            cfg.dist_params['port'] = args.port
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg.work_dir = osp.join(cfg.work_dir, timestamp)
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # init the logger before other steps
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([('{}: {}'.format(k, v))
                          for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('Config:\n{}'.format(cfg.pretty_text))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            args.seed, args.deterministic))
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    if args.pretrained is not None:
        assert isinstance(args.pretrained, str)
        cfg.model.pretrained = args.pretrained
    model = build_model(cfg.model)
    logger.info(model)
    
    datasets = [build_dataset(cfg.data.train)]
    assert len(cfg.workflow) == 1, "Validation is called by hook."
    if cfg.checkpoint_config is not None:
        # save openselfsup version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            openselfsup_version=__version__, config=cfg.text)
            
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)
        
    # add an attribute for visualization convenience
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        timestamp=timestamp,
        meta=meta)

    convert_whole_folder(cfg.work_dir, prefix='mmseg_')
    # fine tuning command
    # ft_cmd = rf'python ~/code/spacenet6/tools/train.py configs/deeplabv3/deeplabv3_r18-d8-selfsup_512x512_20k_sn6_sar_pro_ft.py --options model.pretrained={osp.abspath(osp.join(cfg.work_dir, "mmseg_epoch_200.pth"))}'
    # system(ft_cmd)

if __name__ == '__main__':
    main()
