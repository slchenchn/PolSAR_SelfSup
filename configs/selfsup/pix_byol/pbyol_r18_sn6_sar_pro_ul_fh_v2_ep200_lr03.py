'''
Author: Shuailin Chen
Created Date: 2021-09-10
Last Modified: 2021-10-09
	content: compared with v1, add color jitter
'''

_base_ = ['../_base_/default_runtime.py', 
        '../_base_/models/r18-d8.py', 
        '../_base_/datasets/sn6_sar_pro_ul_fh_v2.py'
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
    dict(type='BYOLHook', end_momentum=1., add_to_tb=True)
]

# optimizer
# optimizer = dict(type='LARS', lr=0.3, weight_decay=0.000001, 
#                 momentum=0.9,
#                 paramwise_options={
#                     '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
#                     'bias': dict(weight_decay=0., lars_exclude=True)})
optimizer = dict(type='SGD', lr=0.3, weight_decay=0.0001, momentum=0.9)
                    
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=2,
    warmup_ratio=0.0001,    # start lr = base_lr * warmup_ratio
    warmup_by_epoch=True)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 200