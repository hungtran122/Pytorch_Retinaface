from __future__ import print_function
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from configs import cfg_mnet, cfg_re50, cfg_re101, cfg_se_resnext101_32x4d
from utils.auto_annotation import *
import cv2
from models.retinaface import RetinaFace

import time
import glob
import shutil

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
					type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.6, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.01, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--run_video', action="store_true", default=False, help='enable running annotation over videos')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
	ckpt_keys = set(pretrained_state_dict.keys())
	model_keys = set(model.state_dict().keys())
	used_pretrained_keys = model_keys & ckpt_keys
	unused_pretrained_keys = ckpt_keys - model_keys
	missing_keys = model_keys - ckpt_keys
	print('Missing keys:{}'.format(len(missing_keys)))
	print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
	print('Used keys:{}'.format(len(used_pretrained_keys)))
	assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
	return True


def remove_prefix(state_dict, prefix):
	''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
	print('remove prefix \'{}\''.format(prefix))
	f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
	return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
	print('Loading pretrained model from {}'.format(pretrained_path))
	if load_to_cpu:
		pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
	else:
		device = torch.cuda.current_device()
		pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
	if "state_dict" in pretrained_dict.keys():
		pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
	else:
		pretrained_dict = remove_prefix(pretrained_dict, 'module.')
	check_keys(model, pretrained_dict)
	model.load_state_dict(pretrained_dict, strict=False)
	return model


if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	torch.set_grad_enabled(False)
	cfg = None
	if args.network == "mobile0.25":
		cfg = cfg_mnet
	elif args.network == "resnet50":
		cfg = cfg_re50
	cfgs = [cfg_re50, cfg_re101, cfg_se_resnext101_32x4d]
	# trained_models = ['./weights/Resnet50_Final.pth', './weights/Resnet101_Final.pth']
	trained_models = ['./weights/Resnet50_Final.pth', './weights/Resnet101_Final.pth', './weights/se_resnext101_32x4d_Final.pth']

	nets = []
	# net and model
	for cfg, trained_model in zip(cfgs,trained_models) :
		net = RetinaFace(cfg=cfg, phase = 'test')
		net = load_model(net, trained_model, args.cpu)
		net.eval()
		print('Finished loading model!')
		# print(net)
		cudnn.benchmark = True
		device = torch.device("cpu" if args.cpu else "cuda")
		net = net.to(device)
		nets.append(net)
	resize = 1
	# testing begin
	# det_dir = 'data/wider_face/images/val/1--Handshaking/'
	with open('folder_list.txt', 'r') as f:
		lines = f.readlines()
	if os.path.exists('det_results'):
		shutil.rmtree('det_results', ignore_errors=True)
	os.makedirs('det_results', exist_ok=True)
	for _dir in lines:
		if args.run_video:
			print ('Running over videos')
			do_annotation_over_video(args, _dir, device, nets, resize, cfg)
		else:
			print('Running over dir')
			modify_annotation(args, _dir, device, nets, resize, cfg)
