'''
Author: Shuailin Chen
Created Date: 2021-09-10
Last Modified: 2021-10-29
	content: byol with filter
'''

_base_ = ['../_base_/default_runtime.py', 
        '../_base_/models/r50-d32.py', 
        '../_base_/datasets/sn6_sar_pro_extend_fh_v2.py',
        '../_base_/schedules/lars_lr02_ep200.py'
        ]

# model settings, output stride=8 for deeplabv3
model = dict(
    type='BF',
    pret_rained=None,
    basemomentum=0.996,
    neck=dict(
        type='NonLinearNeckV2',
        # olive-shaped projector
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        ),
    head=dict(type='LatentPredictHead',
              size_average=True,
              predictor=dict(type='NonLinearNeckV2',
                             in_channels=256, hid_channels=4096,
                             out_channels=256))
    
)
    
# additional hooks
update_interval=1
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., add_to_tb=True, update_interval=update_interval)
]

data = dict(
    imgs_per_gpu=64,
    min_intersect=0.5,  # TODO: find a more elegent way to adjust to 
                        # remote sensing images
)

optimizer_config = dict(update_interval=update_interval)