import os
import numpy as np
import glob
import cv2
import torch
from utils.box_utils import decode, decode_landm
import time
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms, customized_cpu_nms
import shutil

def write_csv(tag_file, boxes, file_name):
	def _bbox_to_coco_bbox(bbox):
		return [(bbox[0]), (bbox[1]),
		        (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]
	with open(tag_file, 'a+') as f:
		row = [file_name + " "]
		for box_id, bbox in enumerate(boxes):
			sc = bbox[4]
			cls_id = bbox[5]
			coco_box = _bbox_to_coco_bbox(bbox[:4].tolist())
			coco_box = np.asarray(coco_box).astype(int)
			coco_box = np.clip(coco_box, a_min=0, a_max=99999)
			coco_box = coco_box.tolist()
			# for tag.csv file
			row.append(str(box_id))
			row.append(" ")
			row.append(str(int(coco_box[0])))
			row.append(" ")
			row.append(str(int(coco_box[1])))
			row.append(" ")
			row.append(str(int(coco_box[2])))
			row.append(" ")
			row.append(str(int(coco_box[3])))
			row.append(" ")
			row.append(str(int(cls_id)))
			row.append(" ")
			row.append('{:.2}'.format(float(sc)))  # don't know
			row.append(" ")
		row.insert(1, str(boxes.shape[0]) + " ") #number of boxes
		row.append('\n')
		for idx, item in enumerate(row):
			if idx == 0:
				txt = item
			else:
				txt += item
		f.write(txt)
def append_box(tag_file, boxes, line=None):
	def _bbox_to_coco_bbox(bbox):
		return [(bbox[0]), (bbox[1]),
		        (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]
	with open(tag_file, 'a+') as f:
		line = line.split(' ')
		num_obj = int(line[1])
		row = []
		for l in line:
			if l == "\n":
				continue
			else:
				row.append (l + " ")
		for box_id, bbox in enumerate(boxes):
			sc = bbox[4]
			cls_id = 4
			coco_box = _bbox_to_coco_bbox(bbox[:4].tolist())
			coco_box = np.asarray(coco_box).astype(int)
			coco_box = np.clip(coco_box, a_min=0, a_max=99999)
			coco_box = coco_box.tolist()
			# for tag.csv file
			row.append(str(box_id + num_obj))
			row.append(" ")
			row.append(str(int(coco_box[0])))
			row.append(" ")
			row.append(str(int(coco_box[1])))
			row.append(" ")
			row.append(str(int(coco_box[2])))
			row.append(" ")
			row.append(str(int(coco_box[3])))
			row.append(" ")
			row.append(str(int(cls_id)))
			row.append(" ")
			row.append('0')  # don't know
			row.append(" ")
		row[1] = str(num_obj + boxes.shape[0]) + " " #number of boxes
		row.append('\n')
		for idx, item in enumerate(row):
			if idx == 0:
				txt = item
			else:
				txt += item
		f.write(txt)
def do_annotation_over_dir(args, _dir, device, nets, resize, cfg):
	exts = ['jpeg', 'png', 'jpg', 'bmp']
	_dir = _dir.replace('\n', '')
	last_dir = _dir.split ('/')[-1]
	os.makedirs(f'auto_annotation_labels/{last_dir}', exist_ok=True)
	tagfile = os.path.join(f'auto_annotation_labels/{last_dir}', 'auto_tag.csv')
	if os.path.exists(tagfile):
		os.remove(tagfile)
	assert os.path.exists(_dir), f'{_dir} does not exist'
	for image_path in glob.glob (_dir + '/*'):
		img_name = os.path.split (image_path)[-1]
		ext = os.path.splitext (img_name)[-1].split ('.')[-1]
		if ext not in exts:
			continue
		img_raw = cv2.imread (image_path, cv2.IMREAD_COLOR)
		img_raws = [img_raw.copy(), img_raw.copy()]
		img = np.float32 (img_raw)

		im_height, im_width, _ = img.shape
		scale = torch.Tensor ([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
		img -= (104, 117, 123)
		img = img.transpose (2, 0, 1)
		img = torch.from_numpy(img).unsqueeze (0)
		img = img.to (device)
		scale = scale.to (device)

		tic = time.time ()
		all_boxes = []
		all_landms = []
		all_scores = []
		drawn_images = []
		for idx, net in enumerate(nets):
			loc, conf, landms = net (img)  # forward pass
			# print ('net forward time: {:.4f}'.format (time.time () - tic))
			priorbox = PriorBox (cfg, image_size=(im_height, im_width))
			priors = priorbox.forward ()
			priors = priors.to (device)
			prior_data = priors.data
			boxes = decode (loc.data.squeeze (0), prior_data, cfg['variance'])
			boxes = boxes * scale / resize
			boxes = boxes.cpu ().numpy ()
			scores = conf.squeeze (0).data.cpu ().numpy ()[:, 1]
			landms = decode_landm (landms.data.squeeze (0), prior_data, cfg['variance'])
			scale1 = torch.Tensor ([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
			                        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
			                        img.shape[3], img.shape[2]])
			scale1 = scale1.to (device)
			landms = landms * scale1 / resize
			landms = landms.cpu ().numpy ()

			# ignore low scores
			inds = np.where (scores > args.confidence_threshold)[0]
			boxes = boxes[inds]
			landms = landms[inds]
			scores = scores[inds]

			all_boxes.append (boxes)
			all_landms.append (landms)
			all_scores.append (scores)

			# keep top-K before NMS
			order = scores.argsort ()[::-1][:args.top_k]
			boxes = boxes[order]
			landms = landms[order]
			scores = scores[order]

			# do NMS
			dets = np.hstack ((boxes, scores[:, np.newaxis])).astype (np.float32, copy=False)
			keep = py_cpu_nms (dets, args.nms_threshold)
			# keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
			dets = dets[keep, :]
			landms = landms[keep]

			# keep top-K faster NMS
			dets = dets[:args.keep_top_k, :]
			landms = landms[:args.keep_top_k, :]

			dets = np.concatenate ((dets, landms), axis=1)

			# show image
			if args.save_image:
				for b in dets:
					if b[4] < args.vis_thres:
						continue
					text = "{:.4f}".format (b[4])
					b = list (map (int, b))
					cv2.rectangle (img_raws[idx], (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
					cx = b[0]
					cy = b[1] + 12
					# cv2.putText (img_raws[idx], text, (cx, cy),
					#              cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

				# landms
				# cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
				# cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
				# cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
				# cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)

				# cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
				# save image
				if idx == len(nets)-1: # last model in list of models
					# name = os.path.join(os.getcwd (), f"det_results/{img_name}")
					img_to_draw = np.vstack(img_raws)
					# cv2.imwrite (name, cv2.resize(img_to_draw, dsize=None, fx=0.5, fy=0.5))




		all_boxes = np.concatenate(all_boxes, axis=0)
		all_landms = np.concatenate (all_landms, axis=0)
		all_scores = np.concatenate (all_scores, axis=0)
		# keep top-K before NMS
		order = all_scores.argsort ()[::-1][:args.top_k]
		all_boxes = all_boxes[order]
		all_landms = all_landms[order]
		all_scores = all_scores[order]

		# do NMS
		dets = np.hstack ((all_boxes, all_scores[:, np.newaxis])).astype (np.float32, copy=False)
		keep = py_cpu_nms (dets, args.nms_threshold)
		# keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
		dets = dets[keep, :]
		all_landms = all_landms[keep]

		# keep top-K faster NMS
		dets = dets[:args.keep_top_k, :]
		all_landms = all_landms[:args.keep_top_k, :]
		dets = np.concatenate ((dets, all_landms), axis=1)
		bbox = np.hstack([dets[:, :5], np.ones((dets.shape[0], 1))])
		write_csv(tagfile, bbox, file_name=img_name)

		# show image
		if args.save_image:
			for b in dets:
				if b[4] < args.vis_thres:
					continue
				text = "{:.4f}".format (b[4])
				b = list (map (int, b))
				cv2.rectangle (img_raw, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 2)
				cx = b[0]
				cy = b[1] + 12
			# cv2.putText(img_raw, text, (cx, cy),
			# 			cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

			# landms
			# cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
			# cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
			# cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
			# cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)

			# cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
			# save image
			name = os.path.join (os.getcwd (), f"det_results/{img_name}")
			img_to_draw = cv2.resize (img_to_draw, dsize=(int(img_raw.shape[1] * (img_raw.shape[0] / img_to_draw.shape[0])), img_raw.shape[0]))
			img_to_draw = np.hstack([img_to_draw, img_raw])
			cv2.imwrite (name, img_to_draw)

def do_annotation_over_video(args, _dir, device, nets, resize, cfg):
	exts = ['mp4', 'avi']
	_dir = _dir.replace('\n', '')

	for video_path in glob.glob (_dir + '/*'):
		video_name = os.path.split (video_path)[-1]
		ext = os.path.splitext (video_name)[-1].split ('.')[-1]
		if ext not in exts:
			continue
		video_name_no_ext = os.path.splitext (video_name)[0]
		target_dir = f'auto_annotation_labels/{video_name_no_ext}'
		if os.path.exists (target_dir):
			shutil.rmtree (target_dir)
		os.makedirs (target_dir, exist_ok=True)
		tagfile = os.path.join (target_dir, 'tag.csv')

		if os.path.exists (tagfile):
			os.remove (tagfile)
		cam = cv2.VideoCapture (video_path)
		frm_num = 0
		while True:
			_, img_raw = cam.read ()
			if img_raw is None:
				break
			frm_num += 1
			if frm_num % 100 != 0:
				continue
			img_name = video_name + '_' + str(format(frm_num, '06d')) + '.jpg'
			img_raws = [img_raw.copy() for i in range(len(nets))]
			img = np.float32 (img_raw)
			im_height, im_width, _ = img.shape
			scale = torch.Tensor ([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
			img -= (104, 117, 123)
			img = img.transpose (2, 0, 1)
			img = torch.from_numpy(img).unsqueeze (0)
			img = img.to (device)
			scale = scale.to (device)

			tic = time.time ()
			all_boxes = []
			all_scores = []
			drawn_images = []
			for idx, net in enumerate(nets):
				loc, conf, landms = net (img)  # forward pass
				# print ('net forward time: {:.4f}'.format (time.time () - tic))
				priorbox = PriorBox (cfg, image_size=(im_height, im_width))
				priors = priorbox.forward ()
				priors = priors.to (device)
				prior_data = priors.data
				boxes = decode (loc.data.squeeze (0), prior_data, cfg['variance'])
				boxes = boxes * scale / resize
				boxes = boxes.cpu ().numpy ()
				scores = conf.squeeze (0).data.cpu ().numpy ()[:, 1]
				landms = decode_landm (landms.data.squeeze (0), prior_data, cfg['variance'])
				scale1 = torch.Tensor ([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
				                        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
				                        img.shape[3], img.shape[2]])
				scale1 = scale1.to (device)
				landms = landms * scale1 / resize
				landms = landms.cpu ().numpy ()

				# ignore low scores
				inds = np.where (scores > args.confidence_threshold)[0]
				boxes = boxes[inds]
				landms = landms[inds]
				scores = scores[inds]

				# keep top-K before NMS
				order = scores.argsort ()[::-1][:args.top_k]
				boxes = boxes[order]
				landms = landms[order]
				scores = scores[order]

				# do NMS
				dets = np.hstack ((boxes, scores[:, np.newaxis])).astype (np.float32, copy=False)
				keep = py_cpu_nms (dets, args.nms_threshold)
				dets = dets[keep, :]
				landms = landms[keep]

				boxes = boxes[keep]
				all_boxes.append (np.hstack ([boxes, np.full_like (boxes[:, :1], fill_value=idx)]))
				all_scores.append (scores[keep])

				# keep top-K faster NMS
				dets = dets[:args.keep_top_k, :]
				landms = landms[:args.keep_top_k, :]

				dets = np.concatenate ((dets, landms), axis=1)

				# show image
				if args.save_image:
					for b in dets:
						if b[4] < args.vis_thres:
							continue
						text = "{:.4f}".format (b[4])
						b = list (map (int, b))
						cv2.rectangle (img_raws[idx], (b[0], b[1]), (b[2], b[3]), (255, 255, 255), 2)
						cx = b[0]
						cy = b[1] + 12
						# cv2.putText (img_raws[idx], text, (cx, cy),
						#              cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

					# landms
					# cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
					# cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
					# cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
					# cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)

					# cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
					# save image
					if idx == len(nets)-1: # last model in list of models
						# name = os.path.join(os.getcwd (), f"det_results/{img_name}")
						img_to_draw = np.vstack(img_raws)
						# cv2.imwrite (name, cv2.resize(img_to_draw, dsize=None, fx=0.5, fy=0.5))

			all_boxes = np.concatenate(all_boxes, axis=0)
			all_scores = np.concatenate (all_scores, axis=0)
			# keep top-K before NMS
			order = all_scores.argsort ()[::-1][:args.top_k]
			all_boxes = all_boxes[order]
			all_scores = all_scores[order]

			# do NMS
			dets = np.hstack ((all_boxes, all_scores[:, np.newaxis])).astype (np.float32, copy=False)
			keep, dont_care = customized_cpu_nms(dets, args.nms_threshold, total_models=len(nets))
			dets_dc = dets[dont_care, :]
			dets = dets[keep, :]

			# keep top-K faster NMS
			dets = dets[:args.keep_top_k, :]
			dets_dc = dets_dc[:args.keep_top_k, :]
			dets = np.delete(dets, [4], axis=1)
			dets_dc = np.delete (dets_dc, [4], axis=1)
			bbox = np.hstack([dets[:, :5], np.full((dets.shape[0], 1), fill_value = 4)])
			if bbox.shape[0] < 3:
				continue
			bbox_dc = np.hstack([dets_dc[:, :5], np.full((dets_dc.shape[0], 1), fill_value = 10)])
			bbox = np.vstack([bbox, bbox_dc])
			write_csv(tagfile, bbox, file_name=img_name)
			imgfile = f'auto_annotation_labels/{video_name_no_ext}/{img_name}'
			cv2.imwrite (imgfile, img_raw)
			# show image
			# if args.save_image:
			if 0:
				for b in dets:
					if b[4] < args.vis_thres:
						continue
					text = "{:.4f}".format (b[4])
					b = list (map (int, b))
					cv2.rectangle (img_raw, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 2)
					cx = b[0]
					cy = b[1] + 12
					cv2.putText (img_raw, text, (cx, cy),
					             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
				for b in dets_dc:
					if b[4] < args.vis_thres:
						continue
					text = "{:.4f}".format (b[4])
					b = list (map (int, b))
					cv2.rectangle (img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
					cx = b[0]
					cy = b[1] + 12
					cv2.putText(img_raw, text, (cx, cy),
								cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

				# landms
				# cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
				# cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
				# cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
				# cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)

				# cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
				# save image
				name = os.path.join (os.getcwd (), f"det_results/{img_name}")
				img_to_draw = cv2.resize (img_to_draw, dsize=(int(img_raw.shape[1] * (img_raw.shape[0] / img_to_draw.shape[0])), img_raw.shape[0]))
				img_to_draw = np.hstack([img_to_draw, img_raw])
				cv2.imwrite (name, img_to_draw)
def modify_annotation(args, _dir, device, nets, resize, cfg):
	exts = ['jpeg', 'png', 'jpg', 'bmp']
	_dir = _dir.replace('\n', '')
	last_dir = _dir.split ('/')[-1]
	assert os.path.exists (_dir), f'{_dir} does not exist'
	tagfile = os.path.join (_dir, 'tag.csv')
	if os.path.exists (tagfile):
		os.remove (tagfile)
	with open(os.path.join (_dir, 'tag_licenseplate.csv'), 'r') as f:
		lines = f.readlines()
	for line in lines:
		img_name = line.split(' ')[0]
		ext = os.path.splitext (img_name)[-1].split ('.')[-1]
		if ext not in exts:
			continue
		img_raw = cv2.imread (os.path.join(_dir, img_name), cv2.IMREAD_COLOR)
		img_raws = [img_raw.copy () for i in range (len (nets))]
		img = np.float32 (img_raw)

		im_height, im_width, _ = img.shape
		scale = torch.Tensor ([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
		img -= (104, 117, 123)
		img = img.transpose (2, 0, 1)
		img = torch.from_numpy(img).unsqueeze (0)
		img = img.to (device)
		scale = scale.to (device)

		tic = time.time ()
		all_boxes = []
		all_landms = []
		all_scores = []
		drawn_images = []
		for idx, net in enumerate (nets):
			loc, conf, landms = net (img)  # forward pass
			# print ('net forward time: {:.4f}'.format (time.time () - tic))
			priorbox = PriorBox (cfg, image_size=(im_height, im_width))
			priors = priorbox.forward ()
			priors = priors.to (device)
			prior_data = priors.data
			boxes = decode (loc.data.squeeze (0), prior_data, cfg['variance'])
			boxes = boxes * scale / resize
			boxes = boxes.cpu ().numpy ()
			scores = conf.squeeze (0).data.cpu ().numpy ()[:, 1]
			landms = decode_landm (landms.data.squeeze (0), prior_data, cfg['variance'])
			scale1 = torch.Tensor ([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
			                        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
			                        img.shape[3], img.shape[2]])
			scale1 = scale1.to (device)
			landms = landms * scale1 / resize
			landms = landms.cpu ().numpy ()

			# ignore low scores
			inds = np.where (scores > args.confidence_threshold)[0]
			boxes = boxes[inds]
			landms = landms[inds]
			scores = scores[inds]

			# keep top-K before NMS
			order = scores.argsort ()[::-1][:args.top_k]
			boxes = boxes[order]
			landms = landms[order]
			scores = scores[order]

			# do NMS
			dets = np.hstack ((boxes, scores[:, np.newaxis])).astype (np.float32, copy=False)
			keep = py_cpu_nms (dets, args.nms_threshold)
			dets = dets[keep, :]
			landms = landms[keep]

			boxes = boxes[keep]
			all_boxes.append (np.hstack ([boxes, np.full_like (boxes[:, :1], fill_value=idx)]))
			all_scores.append (scores[keep])

			# keep top-K faster NMS
			dets = dets[:args.keep_top_k, :]
			landms = landms[:args.keep_top_k, :]

			dets = np.concatenate ((dets, landms), axis=1)

			# show image
			if args.save_image:
				for b in dets:
					if b[4] < args.vis_thres:
						continue
					text = "{:.4f}".format (b[4])
					b = list (map (int, b))
					cv2.rectangle (img_raws[idx], (b[0], b[1]), (b[2], b[3]), (255, 255, 255), 2)
					cx = b[0]
					cy = b[1] + 12
				# cv2.putText (img_raws[idx], text, (cx, cy),
				#              cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

				# landms
				# cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
				# cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
				# cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
				# cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)

				# cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
				# save image
				if idx == len (nets) - 1:  # last model in list of models
					# name = os.path.join(os.getcwd (), f"det_results/{img_name}")
					img_to_draw = np.vstack (img_raws)
			# cv2.imwrite (name, cv2.resize(img_to_draw, dsize=None, fx=0.5, fy=0.5))

		all_boxes = np.concatenate (all_boxes, axis=0)
		all_scores = np.concatenate (all_scores, axis=0)
		# keep top-K before NMS
		order = all_scores.argsort ()[::-1][:args.top_k]
		all_boxes = all_boxes[order]
		all_scores = all_scores[order]

		# do NMS
		dets = np.hstack ((all_boxes, all_scores[:, np.newaxis])).astype (np.float32, copy=False)
		keep, dont_care = customized_cpu_nms (dets, args.nms_threshold, total_models=len (nets))
		dets_dc = dets[dont_care, :]
		dets = dets[keep, :]

		# keep top-K faster NMS
		dets = dets[:args.keep_top_k, :]
		dets_dc = dets_dc[:args.keep_top_k, :]
		dets = np.delete (dets, [4], axis=1)
		dets_dc = np.delete (dets_dc, [4], axis=1)
		bbox = np.hstack ([dets[:, :5], np.full ((dets.shape[0], 1), fill_value=4)])
		# if bbox.shape[0] < 3:
		# 	continue
		bbox_dc = np.hstack ([dets_dc[:, :5], np.full ((dets_dc.shape[0], 1), fill_value=10)])
		bbox = np.vstack ([bbox, bbox_dc])
		append_box(tagfile, bbox, line=line)
		# show image
		if args.save_image:
			for b in dets:
				if b[4] < args.vis_thres:
					continue
				text = "{:.4f}".format (b[4])
				b = list (map (int, b))
				cv2.rectangle (img_raw, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 2)
				cx = b[0]
				cy = b[1] + 12
			# cv2.putText(img_raw, text, (cx, cy),
			# 			cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

			# landms
			# cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
			# cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
			# cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
			# cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)

			# cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
			# save image
			name = os.path.join (os.getcwd (), f"det_results/{img_name}")
			img_to_draw = cv2.resize (img_to_draw, dsize=(int(img_raw.shape[1] * (img_raw.shape[0] / img_to_draw.shape[0])), img_raw.shape[0]))
			img_to_draw = np.hstack([img_to_draw, img_raw])
			cv2.imwrite (name, img_to_draw)