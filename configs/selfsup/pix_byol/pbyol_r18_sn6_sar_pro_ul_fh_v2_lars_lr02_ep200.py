'''
Author: Shuailin Chen
Created Date: 2021-09-10
Last Modified: 2021-10-14
	content: compared with v1, add color jitter
'''

_base_ = ['../_base_/default_runtime.py', 
        '../_base_/models/r18-d8.py', 
        '../_base_/datasets/sn6_sar_pro_ul_fh_v2.py',
        '../_base_/schedules/lars_lr02_ep200.py'
        ]

# model settings, output stride=8 for deeplabv3
model = dict(
    type='PixBYOL',
    base_momentum=0.996,
    neck=dict(
        type='NonLinear1x1ConvNeck',
        # olive-shaped projector
        in_channels=512,
        hid_channels=1024,
        out_channels=64,
        ),
    head=dict(type='PixPredHead',
              size_average=True,
              predictor=dict(type='NonLinear1x1ConvNeck',
                             in_channels=64, hid_channels=1024,
                             out_channels=64))
)
    
# additional hooks
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., add_to_tb=True, update_interval=4)
]

data = dict(
    imgs_per_gpu=64,
)