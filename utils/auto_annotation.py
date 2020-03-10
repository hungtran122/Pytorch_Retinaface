import os
import numpy as np
import glob
import cv2
import torch
from utils.box_utils import decode, decode_landm
import time
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms

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
def do_annotation_over_dir(args, _dir, device, nets, resize, cfg):
	exts = ['jpeg', 'png', 'jpg', 'bmp']
	_dir = _dir.replace('\n', '')
	last_dir = _dir.split ('/')[-1]
	os.makedirs(f'auto_annotation_labels/{last_dir}', exist_ok=True)
	tagfile = os.path.join(f'auto_annotation_labels/{last_dir}', 'auto_tag.csv')
	if os.path.exists(tagfile):
		os.remove(tagfile)
	for image_path in glob.glob (_dir + '/*'):
		img_name = os.path.split (image_path)[-1]
		ext = os.path.splitext (img_name)[-1].split ('.')[-1]
		if ext not in exts:
			continue
		img_raw = cv2.imread (image_path, cv2.IMREAD_COLOR)
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
		for net in nets:
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
			all_boxes.append(boxes)
			all_landms.append(landms)
			all_scores.append(scores)
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
				cv2.rectangle (img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
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
			cv2.imwrite (name, img_raw)