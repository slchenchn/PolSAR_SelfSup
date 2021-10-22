'''
Author: Shuailin Chen
Created Date: 2021-10-22
Last Modified: 2021-10-22
	content: old implement version of ResNet
'''

import warnings
import torch
from torch import nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.runner import Sequential

from ..builder import BACKBONES
from .resnet import ResNet, BasicBlock, Bottleneck


class ResLayerDropPath(Sequential):
	"""ResLayer with drop path (stochastic depth)
	"""

	def __init__(self,
					block,
					inplanes,
					planes,
					num_blocks,
					drop_path_rate,
					stride=1,
					dilation=1,
					avg_down=False,
					conv_cfg=None,
					norm_cfg=dict(type='BN'),
					multi_grid=None,
					contract_dilation=False,
					**kwargs):
		self.block = block

		assert len(drop_path_rate) == num_blocks
		downsample = None
		if stride != 1 or inplanes != planes * block.expansion:
			downsample = []
			conv_stride = stride
			if avg_down:
				conv_stride = 1
				downsample.append(
					nn.AvgPool2d(
						kernel_size=stride,
						stride=stride,
						ceil_mode=True,
						count_include_pad=False))
			downsample.extend([
				build_conv_layer(
					conv_cfg,
					inplanes,
					planes * block.expansion,
					kernel_size=1,
					stride=conv_stride,
					bias=False),
				build_norm_layer(norm_cfg, planes * block.expansion)[1]
			])
			downsample = nn.Sequential(*downsample)

		layers = []
		if multi_grid is None:
			if dilation > 1 and contract_dilation:
				first_dilation = dilation // 2
			else:
				first_dilation = dilation
		else:
			first_dilation = multi_grid[0]
		layers.append(
			block(
				inplanes=inplanes,
				planes=planes,
				stride=stride,
				dilation=first_dilation,
				downsample=downsample,
				conv_cfg=conv_cfg,
				norm_cfg=norm_cfg,
				drop_path_rate = drop_path_rate[0],
				**kwargs))
		inplanes = planes * block.expansion
		for i in range(1, num_blocks):
			layers.append(
				block(
					inplanes=inplanes,
					planes=planes,
					stride=1,
					dilation=dilation if multi_grid is None else multi_grid[i],
					conv_cfg=conv_cfg,
					norm_cfg=norm_cfg,
					drop_path_rate = drop_path_rate[i],
					**kwargs))
		super().__init__(*layers)


class BasicBlockDropPath(BasicBlock):
	''' Basic block with drop path (stochastic depth) '''
	def __init__(self, *args, drop_path_rate=0.1, with_cp=False, **kargs):
		super().__init__(*args, with_cp=with_cp, **kargs)
		self.drop_path_rate=0.1
		self.drop = build_dropout(dict(type='DropPath',
										drop_prob=drop_path_rate))
		self.with_cp = with_cp

	def forward(self, x):
		
		def _inner_forward(x):
			identity = x

			out = self.conv1(x)
			out = self.norm1(out)
			out = self.relu(out)

			out = self.conv2(out)
			out = self.norm2(out)

			if self.downsample is not None:
				identity = self.downsample(x)

			out = self.drop(out)
			out += identity

			return out

		if self.with_cp and x.requires_grad:
			out = cp.checkpoint(_inner_forward, x)
		else:
			out = _inner_forward(x)

		out = self.relu(out)

		return out

	def __repr__(self):
		s = super().__repr__()
		s += f'\ndrop path ratio={self.drop_path_rate}'
		return s


class BottleneckDropPath(Bottleneck):
	''' Bottleneck block with drop path (stochastic depth) '''
	def __init__(self, *args, drop_path_rate=0.1, **kargs):
		super().__init__(*args, **kargs)
		self.drop_path_rate=0.1
		self.drop = build_dropout(dict(type='DropPath',
										drop_prob=drop_path_rate))
	
	def forward(self, x):
		"""Forward function."""

		def _inner_forward(x):
			identity = x

			out = self.conv1(x)
			out = self.norm1(out)
			out = self.relu(out)

			if self.with_plugins:
				out = self.forward_plugin(out, self.after_conv1_plugin_names)

			out = self.conv2(out)
			out = self.norm2(out)
			out = self.relu(out)

			if self.with_plugins:
				out = self.forward_plugin(out, self.after_conv2_plugin_names)

			out = self.conv3(out)
			out = self.norm3(out)

			if self.with_plugins:
				out = self.forward_plugin(out, self.after_conv3_plugin_names)

			if self.downsample is not None:
				identity = self.downsample(x)

			out = self.drop(out)
			out += identity

			return out

		if self.with_cp and x.requires_grad:
			out = cp.checkpoint(_inner_forward, x)
		else:
			out = _inner_forward(x)

		out = self.relu(out)

		return out

	def __repr__(self):
		s = super().__repr__()
		s += f'\ndrop path ratio={self.drop_path_rate}'
		return s

		
@BACKBONES.register_module()
class ResNetDropPath(ResNet):
	''' ResNet with drop path (stochastic depth) '''

	arch_settings = {
        18: (BasicBlockDropPath, (2, 2, 2, 2)),
        34: (BasicBlockDropPath, (3, 4, 6, 3)),
        50: (BottleneckDropPath, (3, 4, 6, 3)),
        101: (BottleneckDropPath, (3, 4, 23, 3)),
        152: (BottleneckDropPath, (3, 8, 36, 3))
    }

	def __init__(self,
				depth,
				in_channels=3,
				stem_channels=64,
				num_stages=4,
				strides=(1, 2, 2, 2),
				dilations=(1, 1, 1, 1),
				out_indices=(0, 1, 2, 3, 4),
				style='pytorch',
				deep_stem=False,
				frozen_stages=-1,
				conv_cfg=None,
				norm_cfg=dict(type='BN', requires_grad=True),
				norm_eval=False,
				with_cp=False,
				zero_init_residual=False,
				drop_path_rate=0.3,
				):
		super(ResNet, self).__init__()
		if depth not in self.arch_settings:
			raise KeyError('invalid depth {} for resnet'.format(depth))
		self.depth = depth
		self.stem_channels = stem_channels
		self.num_stages = num_stages
		assert num_stages >= 1 and num_stages <= 4
		self.strides = strides
		self.dilations = dilations
		assert len(strides) == len(dilations) == num_stages
		self.out_indices = out_indices
		assert max(out_indices) < num_stages + 1
		self.style = style
		self.deep_stem = deep_stem
		self.frozen_stages = frozen_stages
		self.conv_cfg = conv_cfg
		self.norm_cfg = norm_cfg
		self.with_cp = with_cp
		self.norm_eval = norm_eval
		self.zero_init_residual = zero_init_residual
		self.block, stage_blocks = self.arch_settings[depth]
		self.stage_blocks = stage_blocks[:num_stages]
		self.inplanes = 64

		self._make_stem_layer(in_channels, stem_channels)

		# stochastic depth decay rule
		dpr = [
			x.item() for x in torch.linspace(0, drop_path_rate, sum(self.stage_blocks))
		]  

		self.res_layers = []
		for i, num_blocks in enumerate(self.stage_blocks):
			stride = strides[i]
			dilation = dilations[i]
			planes = 64 * 2**i
			res_layer = self.make_res_layer(
				block=self.block,
				inplanes=self.inplanes,
				planes=planes,
				num_blocks=num_blocks,
				stride=stride,
				dilation=dilation,
				style=self.style,
				with_cp=with_cp,
				conv_cfg=conv_cfg,
				drop_path_rate=dpr[:num_blocks],
				)
			dpr = dpr[num_blocks:]
			self.inplanes = planes * self.block.expansion
			layer_name = 'layer{}'.format(i + 1)
			self.add_module(layer_name, res_layer)
			self.res_layers.append(layer_name)

		self._freeze_stages()

		self.feat_dim = self.block.expansion * 64 * 2**(
			len(self.stage_blocks) - 1)

	def make_res_layer(self, **kwargs):
		"""Pack all blocks in a stage into a ``ResLayer``."""
		return ResLayerDropPath(**kwargs)


@BACKBONES.register_module()
class ResNetDropPathV1c(ResNetDropPath):
    """ResNetV1c variant described in [1]_.

    Compared with default ResNet(ResNetV1b), ResNetV1c replaces the 7x7 conv in
    the input stem with three 3x3 convs. For more details please refer to `Bag
    of Tricks for Image Classification with Convolutional Neural Networks
    <https://arxiv.org/abs/1812.01187>`_.
    """

    def __init__(self, **kwargs):
        super().__init__(
            deep_stem=True, **kwargs)