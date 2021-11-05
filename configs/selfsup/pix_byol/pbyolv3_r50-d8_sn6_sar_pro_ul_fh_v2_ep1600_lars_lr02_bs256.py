'''
Author: Shuailin Chen
Created Date: 2021-09-10
Last Modified: 2021-11-05
	content: pbyol v2, replace 1x1 convs of nonlinear neck with 3x3 convs
'''

_base_ = ['../_base_/default_runtime.py', 
        '../_base_/models/r50-d8.py', 
        '../_base_/datasets/sn6_sar_pro_ul_fh_v2.py',
        '../_base_/schedules/lars_lr02_ep1600.py'
        ]

# model settings, output stride=8 for deeplabv3
model = dict(
    type='PixBYOL',
    base_momentum=0.996,
    neck=dict(
        type='NonLinearConvNeck',
        kernel_size=3,
        # olive-shaped projector
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        ),
    head=dict(type='PixPredHead',
              size_average=True,
              predictor=dict(type='NonLinearConvNeck',
                            kernel_size=3,
                            in_channels=256,
                            hid_channels=4096,
                            out_channels=256))
)
    
# additional hooks
update_interval=4
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., add_to_tb=True, update_interval=update_interval)
]

optimizer_config = dict(update_interval=update_interval)

data = dict(
    imgs_per_gpu=32,
)