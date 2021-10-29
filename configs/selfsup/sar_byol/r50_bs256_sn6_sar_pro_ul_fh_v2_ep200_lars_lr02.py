'''
Author: Shuailin Chen
Created Date: 2021-09-08
Last Modified: 2021-10-29
	content: 
'''

_base_ = ['../_base_/default_runtime.py', 
        '../_base_/models/r50-d32.py', 
        '../_base_/datasets/sn6_sar_pro_ul_fh_v2.py',
        '../_base_/schedules/lars_lr02_ep200.py'
        ]

# model settings
model = dict(
    type='BYOL',
    pretrained=None,
    base_momentum=0.996,
    neck=dict(
        type='NonLinearNeckV2',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        with_avg_pool=True),
    head=dict(type='LatentPredictHead',
              size_average=True,
              predictor=dict(type='NonLinearNeckV2',
                             in_channels=256, hid_channels=4096,
                             out_channels=256, with_avg_pool=False)))

data=dict(
    # _delete_=True,
    imgs_per_gpu=64,  # total 32*8=256
    min_intersect=0.5,
    # workers_per_gpu=12,
)

# additional hooks
update_interval=4
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=update_interval)
]

optimizer_config = dict(update_interval=update_interval)
