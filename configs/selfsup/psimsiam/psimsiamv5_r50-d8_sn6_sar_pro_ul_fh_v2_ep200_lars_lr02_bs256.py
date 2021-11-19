'''
Author: Shuailin Chen
Created Date: 2021-09-10
Last Modified: 2021-11-18
	content: compared with v1, add color jitter
'''

_base_ = ['../_base_/default_runtime.py', 
        '../_base_/models/r50-d8.py', 
        '../_base_/datasets/sn6_sar_pro_ul_fh_v2.py',
        '../_base_/schedules/lars_lr02_ep200.py'
        ]

# model settings, output stride=8 for deeplabv3
model = dict(
    type='PixBYOLV5',
    base_momentum=0.0,
    neck=dict(
        type='NonLinear1x1ConvNeck',
        # olive-shaped projector
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        ),
    head=dict(type='PixPredHeadV5',
              size_average=True,
              predictor=dict(type='NonLinear1x1ConvNeck',
                             in_channels=256, hid_channels=4096,
                             out_channels=256))
)
    
# additional hooks
update_interval=8
custom_hooks = [
    dict(type='BYOLHook', end_momentum=0., add_to_tb=True, update_interval=update_interval)
]

optimizer_config = dict(update_interval=update_interval)

data = dict(
    imgs_per_gpu=32,
)