'''
Author: Shuailin Chen
Created Date: 2021-09-08
Last Modified: 2021-10-13
	content: 
'''

from copy import deepcopy

_base_ = ['../_base_/default_runtime.py', 
        # '../_base_/models/r18-d8.py', 
        '../_base_/datasets/sn6_sar_pro_ul_fh_v2.py',
        '../_base_/schedules/lars_lr02_ep200.py'
        ]

# model settings
model = dict(
    type='BYOL',
    pretrained=None,
    base_momentum=0.996,
    backbone=dict(
        # _delete_=True,
        type='ResNetV1c',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')), 
    neck=dict(
        type='NonLinearNeckV2',
        in_channels=512,
        hid_channels=1024,
        out_channels=64,
        with_avg_pool=True),
    head=dict(type='LatentPredictHead',
              size_average=True,
              predictor=dict(type='NonLinearNeckV2',
                             in_channels=64, hid_channels=1024,
                             out_channels=64, with_avg_pool=False)))

data=dict(
    # _delete_=True,
    imgs_per_gpu=64,  # total 32*8=256
    min_intersect=0.5,
    # workers_per_gpu=12,
)

# additional hooks
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=4)
]

