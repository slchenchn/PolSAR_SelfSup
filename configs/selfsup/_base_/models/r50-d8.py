'''
Author: Shuailin Chen
Created Date: 2021-09-30
Last Modified: 2021-10-26
	content: 
'''

_base_ = ['./r50-d32.py']

model = dict(
    backbone=dict(
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        ),
)