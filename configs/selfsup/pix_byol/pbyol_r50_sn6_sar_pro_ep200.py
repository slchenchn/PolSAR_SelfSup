'''
Author: Shuailin Chen
Created Date: 2021-09-10
Last Modified: 2021-09-22
	content: 
'''

from copy import deepcopy
_base_ = '../../base.py'

# model settings, output stride=8 for deeplabv3
model = dict(
    type='PixBYOL',
    pretrained=None,
    base_momentum=0.996,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN'),
        # set output stride=8
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        ),
    neck=dict(
        type='NonLinearNeckV2',
        # olive-shaped projector
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        with_avg_pool=False),
    head=dict(type='PixPredHead',
              size_average=True,
              predictor=dict(type='NonLinearNeckV2',
                             in_channels=256, hid_channels=4096,
                             out_channels=256, with_avg_pool=False)))
                             
# dataset settings
data_source_cfg = dict(
    root = 'data',
    img_dir = 'SN6_full/SAR-PRO',
    ann_dir = 'SN6_sup/slic_mask',
    type='SpaceNet6',
    memcached=False,
    return_label=False,
)
data_train_list = ['data/SN6_full/train.txt', 
                    'data/SN6_full/test.txt']
dataset_type = 'PixBYOLDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    # dict(
    #     type='RandomAppliedTrans',
    #     transforms=[
    #         dict(
    #             type='ColorJitter',
    #             brightness=0.4,
    #             contrast=0.4,
    #             saturation=0.2,
    #             hue=0.1)
    #     ],
    #     p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTransOnlyImg',
        transforms=[
            dict(
                type='BoxBlur',
                radius_min=0,
                radius_max=4,
                )
        ],
        p=1.),
    # dict(type='RandomAppliedTrans',
    #      transforms=[dict(type='Solarization')], p=0.),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), 
                            dict(type='Normalize', **img_norm_cfg)])
train_pipeline1 = deepcopy(train_pipeline)
train_pipeline2 = deepcopy(train_pipeline)
train_pipeline2[3]['p'] = 0.1 # box blur TODO: add gaussian blur
# train_pipeline2[5]['p'] = 0.2 # solarization
    
data = dict(
    imgs_per_gpu=32,  # total 32*8
    workers_per_gpu=12,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list,
            **data_source_cfg),
        pipeline1=train_pipeline1,
        pipeline2=train_pipeline2,
        prefetch=prefetch,
    ))
    
# additional hooks
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1.)
]

# optimizer
optimizer = dict(type='LARS', lr=0.3, weight_decay=0.000001, 
                momentum=0.9,
                paramwise_options={
                    '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
                    'bias': dict(weight_decay=0., lars_exclude=True)})
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