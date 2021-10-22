'''
Author: Shuailin Chen
Created Date: 2021-09-30
Last Modified: 2021-10-22
	content: resnet with stochastic path
'''

_base_=['./r18-d8.py']

model = dict(
    backbone=dict(
        type='ResNetDropPathV1c',
        drop_path_rate=0.3
        ),
)