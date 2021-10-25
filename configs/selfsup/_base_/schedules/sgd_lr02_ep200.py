'''
Author: Shuailin Chen
Created Date: 2021-10-22
Last Modified: 2021-10-25
	content: 
'''


# optimizer
# optimizer = dict(type='LARS', lr=0.3, weight_decay=0.000001, 
#                 momentum=0.9,
#                 paramwise_options={
#                     '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
#                     'bias': dict(weight_decay=0., lars_exclude=True)})
optimizer = dict(type='SGD', lr=0.2, weight_decay=0.0000015, momentum=0.9)
                    
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=2,
    warmup_ratio=0.0001,    # start lr = base_lr * warmup_ratio
    warmup_by_epoch=True)
checkpoint_config = dict(interval=20)

# runtime settings
total_epochs = 200