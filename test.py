#!/usr/bin/env python
# coding: utf-8

import argparse
import csv
import os
import os.path as osp
import time
from glob import glob

# from mean_average_precision.detection_map import DetectionMAP
import numpy as np
import torch
from tqdm import tqdm

from head_detection.data import (cfg_mnet, cfg_res50, cfg_res50_4fpn,
                                 cfg_res152, ch_anchors, combined_anchors,
                                 headhunt_anchors, sh_anchors)
from head_detection.models.head_detect import customRCNN
from head_detection.utils import get_state_dict, plot_ims, to_torch
from head_detection.vision.utils import init_distributed_mode

try:
    from scipy.misc import imread, imsave
except ImportError:
    from scipy.misc.pilutil import imread

parser = argparse.ArgumentParser(description='Testing script')
parser.add_argument('--test_dataset', help='Dataset .txt file')
parser.add_argument('--pretrained_model', help='resume net for retraining')
parser.add_argument('--plot_folder', help='Location to plot results on images')

parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

parser.add_argument('--benchmark', default='Combined', help='Benchmark for training/validation')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--min_size', default=800, type=int, help='Optionally plot first N images of test')
parser.add_argument('--max_size', default=1400, type=int, help='Optionally plot first N images of test')

parser.add_argument('--ext', default='.jpg', type=str, help='Image file extensions')
parser.add_argument('--outfile', help='Location to save results in mot format')

parser.add_argument('--backbone', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--context', help='Whether to use context model')
parser.add_argument('--use_deform', default=False, type=bool, help='Use Deformable SSH')
parser.add_argument('--det_thresh', default=0.3, type=float, help='Number of workers used in dataloading')
parser.add_argument('--default_filter', default=False, type=bool, help='Use old filters')
parser.add_argument('--soft_nms', default=False, type=bool, help='Use soft nms?')
parser.add_argument('--upscale_rpn', default=False, type=bool, help='Upscale RPN feature maps')

args = parser.parse_args()

##################################
## Set device and config ##########
##################################
if torch.cuda.is_available():
    device = torch.device('cuda')
cfg = None
if args.backbone == "mobile0.25":
    cfg = cfg_mnet
elif args.backbone == "resnet50":
    cfg = cfg_res50_4fpn
elif args.backbone == "resnet152":
    cfg = cfg_res152
else:
    raise ValueError("Invalid configuration")

##########################
# outfile and plot file ##
##########################
if args.plot_folder is not None:
    os.makedirs(osp.join(args.plot_folder, 'img'), exist_ok=True)
    args.plot_folder = osp.join(args.plot_folder, 'img')
else:
    raise AssertionError("Must provide save directory")
if args.outfile is None:
    args.outfile = osp.join(args.plot_folder, 'results.txt')

def fetch_images():
    all_ext = '*'+args.ext
    all_ims = sorted(glob(osp.join(args.test_dataset, all_ext)))
    batched_ims = [all_ims[k:k+args.batch_size] for k in range(0, len(all_ims), args.batch_size)]
    for b_ind, batch in enumerate(batched_ims):
        img_ar = []
        target_ar = []
        for idx, im in enumerate(batch):
            img_ar.extend(to_torch(imread(im)))
            target_ar.append(get_test_dict((args.batch_size*b_ind)+idx+1))
        yield img_ar, target_ar


def get_test_dict(idx):
    """
    Get FRCNN style dict
    """

    num_objs = 0
    boxes = torch.zeros((num_objs, 4), dtype=torch.float32)

    return {'boxes': boxes,
            'labels': torch.ones((num_objs,), dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            'iscrowd': torch.zeros((num_objs,), dtype=torch.int64),
            'visibilities': torch.zeros((num_objs), dtype=torch.float32)}

def create_model(combined_cfg):
    kwargs = {}
    kwargs['min_size'] = args.min_size
    kwargs['max_size'] = args.max_size
    kwargs['box_score_thresh'] = args.det_thresh
    kwargs['box_nms_thresh'] = 0.5
    kwargs['box_detections_per_img'] = 300 # increase max det to max val in our benchmark
    model = customRCNN(cfg=combined_cfg, use_deform=args.use_deform,
                       context=args.context, default_filter=args.default_filter,
                       soft_nms=args.soft_nms, upscale_rpn=args.upscale_rpn,
                       **kwargs).cuda()
    return model

def write_results_files(results):
        files = {}
        for image_id, res in results.items():
            # check if out in keys and create empty list if not
            if args.outfile not in files.keys():
                files[args.outfile] = []

            for box, score in zip(res['boxes'], res['scores']):
                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()
                files[args.outfile].append(
                    [image_id, -1, x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])

        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)

@torch.no_grad()
def test():
    # print("Testing FPN. On single GPU without Parallelism")
    cpu_device = torch.device("cpu")

    # Set benchmark related parameters
    if args.benchmark == 'ScutHead':
        combined_cfg = {**cfg, **sh_anchors}
    elif args.benchmark == 'CHuman':
        combined_cfg = {**cfg, **ch_anchors}
    elif args.benchmark == 'Combined':
        combined_cfg = {**cfg, **combined_anchors}
    else:
        raise ValueError("New dataset has to be registered")
    model = create_model(combined_cfg)
    new_state_dict = get_state_dict(model, args.pretrained_model)
    model.load_state_dict(new_state_dict, strict=True)
    model = model.eval()
    results = {}
    for img_ind, (images, targets) in enumerate(tqdm(fetch_images())):
        np_images = [(ims.cpu().numpy()*255.).astype(np.uint8) for ims in images]
        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        for b_ind, (np_im, out, tar) in enumerate(zip(np_images, outputs, targets)):
            out_dict = {'boxes': out['boxes'].cpu(), 'scores': out['scores'].cpu()}
            results[tar['image_id'].item()] = out_dict
            plot_images = plot_ims(np_im, out['boxes'].cpu().numpy())
            imsave(osp.join(args.plot_folder, str((args.batch_size*img_ind)+b_ind+1) + '.jpg'), plot_images)
    write_results_files(results)


if __name__ == '__main__':
    test()
