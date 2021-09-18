'''
Author: Shuailin Chen
Created Date: 2021-09-10
Last Modified: 2021-09-18
	content: 
'''

_base_ = '../../base.py'

# model settings, output stride=8 for deeplabv3
model = dict(
    type='BYOL',
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
    head=dict(type='LatentPredictHead',
              size_average=True,
              predictor=dict(type='NonLinearNeckV2',
                             in_channels=256, hid_channels=4096,
                             out_channels=256, with_avg_pool=False)))
                             
# dataset settings
data_source_cfg = dict(
    type='SARCD',
    memcached=False,
)
data_train_list = ['data/SN6_full/train.txt', 
                    'data/SN6_full/test.txt']
data_train_root = 'data/SN6_full/SAR-PRO'
dataset_type = 'BYOLDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    # dict(
    #     type='RandomAppliedTrans',
    #     transforms=[
    #         dict(
    #             type='ColorJitter',
    #             brightness=0.8,
    #             contrast=0.8,
    #             saturation=0.8,
    #             hue=0.2)
    #     ],
    #     p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='BoxBlur',
                radius_min=0,
                radius_max=4,
                )
        ],
        p=0.5),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
    
data = dict(
    imgs_per_gpu=32,  # total 32*8
    workers_per_gpu=12,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
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
    warmup_iters=10,
    warmup_ratio=0.0001,    # start lr = base_lr * warmup_ratio
    warmup_by_epoch=True)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 200