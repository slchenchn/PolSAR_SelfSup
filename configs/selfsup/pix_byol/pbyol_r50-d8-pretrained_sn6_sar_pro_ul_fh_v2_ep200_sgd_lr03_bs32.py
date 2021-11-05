'''
Author: Shuailin Chen
Created Date: 2021-09-10
Last Modified: 2021-11-05
	content: compared with v1, add color jitter
'''

_base_ = ['../_base_/default_runtime.py', 
        '../_base_/models/r50-d8.py', 
        '../_base_/datasets/sn6_sar_pro_ul_fh_v2.py',
        '../_base_/schedules/sgd_lr03_ep200.py'
        ]

# model settings, output stride=8 for deeplabv3
model = dict(
    pretrained='open-mmlab://resnet50_v1c',
    type='PixBYOL',
    base_momentum=0.996,
    neck=dict(
        type='NonLinear1x1ConvNeck',
        # olive-shaped projector
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        ),
    head=dict(type='PixPredHead',
              size_average=True,
              predictor=dict(type='NonLinear1x1ConvNeck',
                             in_channels=256, hid_channels=4096,
                             out_channels=256))
)
    
# additional hooks
update_interval=1
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., add_to_tb=True, update_interval=update_interval)
]

optimizer_config = dict(update_interval=update_interval)

data = dict(
    imgs_per_gpu=32,
)