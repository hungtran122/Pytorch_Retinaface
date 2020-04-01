import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH
import torchvision.models as models



class ClassHead(nn.Module):
	def __init__(self,inchannels=512,num_anchors=3):
		super(ClassHead,self).__init__()
		self.num_anchors = num_anchors
		self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

	def forward(self,x):
		out = self.conv1x1(x)
		out = out.permute(0,2,3,1).contiguous()

		return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
	def __init__(self,inchannels=512,num_anchors=3):
		super(BboxHead,self).__init__()
		self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

	def forward(self,x):
		out = self.conv1x1(x)
		out = out.permute(0,2,3,1).contiguous()

		return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
	def __init__(self,inchannels=512,num_anchors=3):
		super(LandmarkHead,self).__init__()
		self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

	def forward(self,x):
		out = self.conv1x1(x)
		out = out.permute(0,2,3,1).contiguous()

		return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
	def __init__(self, cfg = None, phase = 'train'):
		"""
		:param cfg:  Network related settings.
		:param phase: train or test.
		"""
		super(RetinaFace,self).__init__()
		self.phase = phase
		backbone = None
		if cfg['name'] == 'mobilenet0.25':
			backbone = MobileNetV1()
			if cfg['pretrain']:
				checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
				from collections import OrderedDict
				new_state_dict = OrderedDict()
				for k, v in checkpoint['state_dict'].items():
					name = k[7:]  # remove module.
					new_state_dict[name] = v
				# load params
				backbone.load_state_dict(new_state_dict)
		elif cfg['name'] == 'Resnet50':
			backbone = models.resnet50(pretrained=cfg['pretrain'])
		elif cfg['name'] == 'Resnet101':
			backbone = models.resnet101(pretrained=cfg['pretrain'])
		elif cfg['name'] == 'Densenet121':
			backbone = models.densenet121(pretrained=cfg['pretrain']).features
		elif cfg['name'] == 'se_resnext101_32x4d':
			backbone = se_resnext101_32x4d()
		self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
		in_channels_stage2 = cfg['in_channel']
		if 'Resnet' in cfg['name']:
			in_channels_list = [
				in_channels_stage2 * 2,
				in_channels_stage2 * 4,
				in_channels_stage2 * 8,
			]
		elif 'Densenet' in cfg['name']:
			in_channels_list = [
				in_channels_stage2 * 2,
				in_channels_stage2 * 4,
				in_channels_stage2 * 4,
			]
		else:
			in_channels_list = [
				in_channels_stage2 * 2,
				in_channels_stage2 * 4,
				in_channels_stage2 * 8,
			]
		out_channels = cfg['out_channel']
		self.fpn = FPN(in_channels_list,out_channels)
		self.ssh1 = SSH(out_channels, out_channels)
		self.ssh2 = SSH(out_channels, out_channels)
		self.ssh3 = SSH(out_channels, out_channels)

		self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
		self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
		self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

	def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
		classhead = nn.ModuleList()
		for i in range(fpn_num):
			classhead.append(ClassHead(inchannels,anchor_num))
		return classhead

	def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
		bboxhead = nn.ModuleList()
		for i in range(fpn_num):
			bboxhead.append(BboxHead(inchannels,anchor_num))
		return bboxhead

	def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
		landmarkhead = nn.ModuleList()
		for i in range(fpn_num):
			landmarkhead.append(LandmarkHead(inchannels,anchor_num))
		return landmarkhead

	def forward(self,inputs):
		out = self.body(inputs)

		# FPN
		fpn = self.fpn(out)

		# SSH
		feature1 = self.ssh1(fpn[0])
		feature2 = self.ssh2(fpn[1])
		feature3 = self.ssh3(fpn[2])
		features = [feature1, feature2, feature3]

		bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
		classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
		ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

		if self.phase == 'train':
			output = (bbox_regressions, classifications, ldm_regressions)
		else:
			output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
		return output

#####################################################################################################
#####################################################################################################
"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

from collections import OrderedDict
import math

import torch.nn as nn
from torch.utils import model_zoo


__all__ = ['SENet', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
		   'se_resnext50_32x4d', 'se_resnext101_32x4d']

pretrained_settings = {
	'senet154': {
		'imagenet': {
			'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
			'input_space': 'RGB',
			'input_size': [3, 224, 224],
			'input_range': [0, 1],
			'mean': [0.485, 0.456, 0.406],
			'std': [0.229, 0.224, 0.225],
			'num_classes': 1000
		}
	},
	'se_resnet50': {
		'imagenet': {
			'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
			'input_space': 'RGB',
			'input_size': [3, 224, 224],
			'input_range': [0, 1],
			'mean': [0.485, 0.456, 0.406],
			'std': [0.229, 0.224, 0.225],
			'num_classes': 1000
		}
	},
	'se_resnet101': {
		'imagenet': {
			'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
			'input_space': 'RGB',
			'input_size': [3, 224, 224],
			'input_range': [0, 1],
			'mean': [0.485, 0.456, 0.406],
			'std': [0.229, 0.224, 0.225],
			'num_classes': 1000
		}
	},
	'se_resnet152': {
		'imagenet': {
			'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
			'input_space': 'RGB',
			'input_size': [3, 224, 224],
			'input_range': [0, 1],
			'mean': [0.485, 0.456, 0.406],
			'std': [0.229, 0.224, 0.225],
			'num_classes': 1000
		}
	},
	'se_resnext50_32x4d': {
		'imagenet': {
			'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
			'input_space': 'RGB',
			'input_size': [3, 224, 224],
			'input_range': [0, 1],
			'mean': [0.485, 0.456, 0.406],
			'std': [0.229, 0.224, 0.225],
			'num_classes': 1000
		}
	},
	'se_resnext101_32x4d': {
		'imagenet': {
			'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
			'input_space': 'RGB',
			'input_size': [3, 224, 224],
			'input_range': [0, 1],
			'mean': [0.485, 0.456, 0.406],
			'std': [0.229, 0.224, 0.225],
			'num_classes': 1000
		}
	},
}


class SEModule(nn.Module):

	def __init__(self, channels, reduction):
		super(SEModule, self).__init__()

		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
							 padding=0)
		self.relu = nn.ReLU(inplace=True)
		self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
							 padding=0)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		module_input = x
		x = self.avg_pool(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.sigmoid(x)
		return module_input * x


class Bottleneck(nn.Module):
	"""
	Base class for bottlenecks that implements `forward()` method.
	"""
	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out = self.se_module(out) + residual
		out = self.relu(out)

		return out


class SEBottleneck(Bottleneck):
	"""
	Bottleneck for SENet154.
	"""
	expansion = 4

	def __init__(self, inplanes, planes, groups, reduction, stride=1,
				 downsample=None):
		super(SEBottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes * 2)
		self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
							   stride=stride, padding=1, groups=groups,
							   bias=False)
		self.bn2 = nn.BatchNorm2d(planes * 4)
		self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
							   bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.se_module = SEModule(planes * 4, reduction=reduction)
		self.downsample = downsample
		self.stride = stride


class SEResNetBottleneck(Bottleneck):
	"""
	ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
	implementation and uses `stride=stride` in `conv1` and not in `conv2`
	(the latter is used in the torchvision implementation of ResNet).
	"""
	expansion = 4

	def __init__(self, inplanes, planes, groups, reduction, stride=1,
				 downsample=None):
		super(SEResNetBottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
							   stride=stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
							   groups=groups, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.se_module = SEModule(planes * 4, reduction=reduction)
		self.downsample = downsample
		self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
	"""
	ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
	"""
	expansion = 4

	def __init__(self, inplanes, planes, groups, reduction, stride=1,
				 downsample=None, base_width=4):
		super(SEResNeXtBottleneck, self).__init__()
		width = math.floor(planes * (base_width / 64)) * groups
		self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
							   stride=1)
		self.bn1 = nn.BatchNorm2d(width)
		self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
							   padding=1, groups=groups, bias=False)
		self.bn2 = nn.BatchNorm2d(width)
		self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.se_module = SEModule(planes * 4, reduction=reduction)
		self.downsample = downsample
		self.stride = stride


class SENet(nn.Module):

	def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
				 inplanes=128, input_3x3=True, downsample_kernel_size=3,
				 downsample_padding=1, num_classes=1000):
		"""
		Parameters
		----------
		block (nn.Module): Bottleneck class.
			- For SENet154: SEBottleneck
			- For SE-ResNet models: SEResNetBottleneck
			- For SE-ResNeXt models:  SEResNeXtBottleneck
		layers (list of ints): Number of residual blocks for 4 layers of the
			network (layer1...layer4).
		groups (int): Number of groups for the 3x3 convolution in each
			bottleneck block.
			- For SENet154: 64
			- For SE-ResNet models: 1
			- For SE-ResNeXt models:  32
		reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
			- For all models: 16
		dropout_p (float or None): Drop probability for the Dropout layer.
			If `None` the Dropout layer is not used.
			- For SENet154: 0.2
			- For SE-ResNet models: None
			- For SE-ResNeXt models: None
		inplanes (int):  Number of input channels for layer1.
			- For SENet154: 128
			- For SE-ResNet models: 64
			- For SE-ResNeXt models: 64
		input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
			a single 7x7 convolution in layer0.
			- For SENet154: True
			- For SE-ResNet models: False
			- For SE-ResNeXt models: False
		downsample_kernel_size (int): Kernel size for downsampling convolutions
			in layer2, layer3 and layer4.
			- For SENet154: 3
			- For SE-ResNet models: 1
			- For SE-ResNeXt models: 1
		downsample_padding (int): Padding for downsampling convolutions in
			layer2, layer3 and layer4.
			- For SENet154: 1
			- For SE-ResNet models: 0
			- For SE-ResNeXt models: 0
		num_classes (int): Number of outputs in `last_linear` layer.
			- For all models: 1000
		"""
		super(SENet, self).__init__()
		self.inplanes = inplanes
		if input_3x3:
			layer0_modules = [
				('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
									bias=False)),
				('bn1', nn.BatchNorm2d(64)),
				('relu1', nn.ReLU(inplace=True)),
				('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
									bias=False)),
				('bn2', nn.BatchNorm2d(64)),
				('relu2', nn.ReLU(inplace=True)),
				('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
									bias=False)),
				('bn3', nn.BatchNorm2d(inplanes)),
				('relu3', nn.ReLU(inplace=True)),
			]
		else:
			layer0_modules = [
				('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
									padding=3, bias=False)),
				('bn1', nn.BatchNorm2d(inplanes)),
				('relu1', nn.ReLU(inplace=True)),
			]
		# To preserve compatibility with Caffe weights `ceil_mode=True`
		# is used instead of `padding=1`.
		layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
													ceil_mode=True)))
		self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
		self.layer1 = self._make_layer(
			block,
			planes=64,
			blocks=layers[0],
			groups=groups,
			reduction=reduction,
			downsample_kernel_size=1,
			downsample_padding=0
		)
		self.layer2 = self._make_layer(
			block,
			planes=128,
			blocks=layers[1],
			stride=2,
			groups=groups,
			reduction=reduction,
			downsample_kernel_size=downsample_kernel_size,
			downsample_padding=downsample_padding
		)
		self.layer3 = self._make_layer(
			block,
			planes=256,
			blocks=layers[2],
			stride=2,
			groups=groups,
			reduction=reduction,
			downsample_kernel_size=downsample_kernel_size,
			downsample_padding=downsample_padding
		)
		self.layer4 = self._make_layer(
			block,
			planes=512,
			blocks=layers[3],
			stride=2,
			groups=groups,
			reduction=reduction,
			downsample_kernel_size=downsample_kernel_size,
			downsample_padding=downsample_padding
		)
		# self.avg_pool = nn.AvgPool2d(7, stride=1)
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
		self.last_linear = nn.Linear(512 * block.expansion, num_classes)

	def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
					downsample_kernel_size=1, downsample_padding=0):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=downsample_kernel_size, stride=stride,
						  padding=downsample_padding, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, groups, reduction, stride,
							downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups, reduction))

		return nn.Sequential(*layers)

	def features(self, x):
		x = self.layer0(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		return x

	def logits(self, x):
		x = self.avg_pool(x)
		if self.dropout is not None:
			x = self.dropout(x)
		x = x.view(x.size(0), -1)
		x = self.last_linear(x)
		return x

	def forward(self, x):
		x = self.features(x)
		x = self.logits(x)
		return x


def initialize_pretrained_model(model, num_classes, settings):
	assert num_classes == settings['num_classes'], \
		'num_classes should be {}, but is {}'.format(
			settings['num_classes'], num_classes)
	model.load_state_dict(model_zoo.load_url(settings['url']))
	model.input_space = settings['input_space']
	model.input_size = settings['input_size']
	model.input_range = settings['input_range']
	model.mean = settings['mean']
	model.std = settings['std']


def senet154(num_classes=1000, pretrained='imagenet'):
	model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
				  dropout_p=0.2, num_classes=num_classes)
	if pretrained is not None:
		settings = pretrained_settings['senet154'][pretrained]
		initialize_pretrained_model(model, num_classes, settings)
	return model


def se_resnet50(num_classes=1000, pretrained='imagenet'):
	model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
				  dropout_p=None, inplanes=64, input_3x3=False,
				  downsample_kernel_size=1, downsample_padding=0,
				  num_classes=num_classes)
	if pretrained is not None:
		settings = pretrained_settings['se_resnet50'][pretrained]
		initialize_pretrained_model(model, num_classes, settings)
	return model


def se_resnet101(num_classes=1000, pretrained='imagenet'):
	model = SENet(SEResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16,
				  dropout_p=None, inplanes=64, input_3x3=False,
				  downsample_kernel_size=1, downsample_padding=0,
				  num_classes=num_classes)
	if pretrained is not None:
		settings = pretrained_settings['se_resnet101'][pretrained]
		initialize_pretrained_model(model, num_classes, settings)
	return model


def se_resnet152(num_classes=1000, pretrained='imagenet'):
	model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16,
				  dropout_p=None, inplanes=64, input_3x3=False,
				  downsample_kernel_size=1, downsample_padding=0,
				  num_classes=num_classes)
	if pretrained is not None:
		settings = pretrained_settings['se_resnet152'][pretrained]
		initialize_pretrained_model(model, num_classes, settings)
	return model


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
	model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
				  dropout_p=None, inplanes=64, input_3x3=False,
				  downsample_kernel_size=1, downsample_padding=0,
				  num_classes=num_classes)
	if pretrained is not None:
		settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
		initialize_pretrained_model(model, num_classes, settings)
	return model


def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
	model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
				  dropout_p=None, inplanes=64, input_3x3=False,
				  downsample_kernel_size=1, downsample_padding=0,
				  num_classes=num_classes)
	if pretrained is not None:
		settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
		initialize_pretrained_model(model, num_classes, settings)
	return model

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ShakeDropFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, training=True, p_drop=0.5, alpha_range=[-1, 1]):
		if training:
			gate = torch.cuda.FloatTensor([0]).bernoulli_(1 - p_drop)
			ctx.save_for_backward(gate)
			if gate.item() == 0:
				alpha = torch.cuda.FloatTensor(x.size(0)).uniform_(*alpha_range)
				alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x)
				return alpha * x
			else:
				return x
		else:
			return (1 - p_drop) * x

	@staticmethod
	def backward(ctx, grad_output):
		gate = ctx.saved_tensors[0]
		if gate.item() == 0:
			beta = torch.cuda.FloatTensor(grad_output.size(0)).uniform_(0, 1)
			beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
			beta = Variable(beta)
			return beta * grad_output, None, None, None
		else:
			return grad_output, None, None, None


class ShakeDrop(nn.Module):

	def __init__(self, p_drop=0.5, alpha_range=[-1, 1]):
		super(ShakeDrop, self).__init__()
		self.p_drop = p_drop
		self.alpha_range = alpha_range

	def forward(self, x):
		return ShakeDropFunction.apply(x, self.training, self.p_drop, self.alpha_range)

import torch
import torch.nn as nn
import torch.nn.functional as F  #(uncomment if needed,but you likely already have it)

#Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
#https://arxiv.org/abs/1908.08681v1
#implemented for PyTorch / FastAI by lessw2020
#github: https://github.com/lessw2020/mish

class Mish(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, x):
		#inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
		return x *( torch.tanh(F.softplus(x)))
# Drop-Activation
class dropRelu(nn.Module):
	def __init__(self, keep_prob=0.95):
		'''
		:param keep_prob: the probability of retaining the ReLU activation
		'''
		super(dropRelu, self).__init__()
		self.keep_prob = keep_prob
	def forward(self, x):
		'''
		:param x: input of x
		:return: drop activation during training or testing phase
		'''
		size_len = len(x.size())
		if self.training:
			Bernoulli_mask = torch.cuda.FloatTensor(x.size()[0:size_len]).fill_(1)
			Bernoulli_mask.bernoulli_(self.keep_prob)
			temp = torch.Tensor().cuda()
			output = torch.Tensor().cuda()
			temp.resize_as_(x).copy_(x)
			output.resize_as_(x).copy_(x)
			output.mul_(Bernoulli_mask)
			output.mul_(-1)
			output.add_(temp)
			temp.clamp_(min = 0)
			temp.mul_(Bernoulli_mask)
			output.add_(temp)
			return output
		else:
			temp = torch.Tensor().cuda()
			output = torch.Tensor().cuda()
			temp.resize_as_(x).copy_(x)
			output.resize_as_(x).copy_(x)
			output.mul_(self.keep_prob)
			output.mul_(-1)
			output.add_(temp)
			temp.clamp_(min=0)
			temp.mul_(self.keep_prob)
			output.add_(temp)
			return output