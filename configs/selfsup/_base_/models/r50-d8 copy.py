'''
Author: Shuailin Chen
Created Date: 2021-09-30
Last Modified: 2021-09-30
	content: 
'''

model = dict(
    pretrained=None,
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        in_channels=3,
        out_indices=[4],  # x: stage-x + 1
        norm_cfg=dict(type='BN'),
        # set output stride=8
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        ),
)