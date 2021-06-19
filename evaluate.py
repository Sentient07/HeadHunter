
#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os.path as osp
import os
import cv2
# from mean_average_precision.detection_map import DetectionMAP
import torch
import torch.utils.data as data

from head_detection.data import (HeadDataset, cfg_mnet, cfg_res50,
                                 cfg_res50_4fpn, cfg_res152, ch_anchors,
                                 combined_anchors, headhunt_anchors,
                                 sh_anchors, compute_mean_std)
from head_detection.models.head_detect import customRCNN
from head_detection.utils import get_state_dict, restore_network
from head_detection.vision.engine import evaluate
from head_detection.vision.utils import MetricLogger
from head_detection.vision.utils import collate_fn as coco_collate
from head_detection.vision.utils import init_distributed_mode
from tqdm import tqdm
from collections import defaultdict
import numpy as np

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

parser = argparse.ArgumentParser(description='Evaluation script')
parser.add_argument('--test_dataset', required=True, help='Dataset .txt file')
parser.add_argument('--pretrained_model', required=True, help='resume net for retraining')
parser.add_argument('--exp_name', required=True, type=str,
                    help='Name of file to save the test stats')
parser.add_argument('--context', help='Whether to use context model')


parser.add_argument('--backbone', default='resnet50', help='Backbone network mobilenet, resnet50, resnet152')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--n_gpu', default=1, type=int, help='Number of GPUs')

parser.add_argument('--results', type=str, help='Where to save the results as txt')
parser.add_argument('--benchmark', default='Combined', help='Benchmark for training/validation')
parser.add_argument('--base_path', default='/temp_dd/igrida-fs1/rsundara/dataset', help='Base Path for dataset')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--min_size', default=800, type=int, help='If left None, default image size is used')
parser.add_argument('--max_size', default=1400, type=int, help='If left None, default image size is used')
parser.add_argument('--use_deform', default=False, type=bool, help='Use Deformable SSH')
parser.add_argument('--det_thresh', default=0.3, type=float, help='Number of workers used in dataloading')
parser.add_argument('--default_filter', default=False, type=bool, help='Only to be used for HollywoodHeads dataset')
parser.add_argument('--soft_nms', default=False, type=bool, help='Use soft nms?')
parser.add_argument('--upscale_rpn', default=False, type=bool, help='Upscale RPN feature maps')
parser.add_argument('--precmp_mean', default=False, type=bool, help='Dont recompute RGB means')


parser.add_argument('--log_dir', default='', type=str,
                    help='place to log the validation results')


args = parser.parse_args()
log_name = osp.join(args.log_dir, args.exp_name + "_testing.log")
logging.basicConfig(filename=log_name, filemode='w', level=logging.INFO)
logging.info("Writing logs to this file" + str(log_name))
print("Logging into %s" %log_name)

if torch.cuda.is_available():
    device = torch.device('cuda')

median_anchors = False if not args.benchmark else True
print("Using Median anchors " + str(median_anchors))

cfg = cfg_res50_4fpn


@torch.no_grad()
def test():
    cpu_device = torch.device("cpu")
    kwargs = {}
    kwargs['min_size'] = args.min_size
    kwargs['max_size'] = args.max_size
    kwargs['box_score_thresh'] = args.det_thresh

    if args.precmp_mean:
        dataset_mean, dataset_std = compute_mean_std(args.test_dataset, args.base_path)
        print(dataset_mean)
        print(dataset_std)
    else:
        dset_mean_std = [[117, 110, 105], [67.10, 65.45, 66.23]]
        dataset_mean = [i/255. for i in dset_mean_std[0]]
        dataset_std = [i/255. for i in dset_mean_std[1]]
    kwargs['image_mean'] = dataset_mean
    kwargs['image_std'] = dataset_std
    # kwargs['box_nms_thresh'] = 0.5
    kwargs['box_detections_per_img'] = 300 # increase max det to max val in our benchmark
    # Set benchmark related parameters
    if args.benchmark == 'ScutHead':
        combined_cfg = {**cfg, **sh_anchors}
    elif args.benchmark == 'CHuman':
        combined_cfg = {**cfg, **ch_anchors}
    elif args.benchmark == 'Combined':
        combined_cfg = {**cfg, **combined_anchors}
    else:
        raise ValueError("New dataset has to be registered")

    model = customRCNN(cfg=combined_cfg, use_deform=args.use_deform,
                       context=args.context, default_filter=args.default_filter,
                       soft_nms=args.soft_nms, upscale_rpn=args.upscale_rpn,
                       median_anchors=median_anchors,
                       **kwargs).cuda().eval()
    model = restore_network(model, args.pretrained_model)
    model_without_ddp = model
    if args.test_dataset == 'all':
        test_path = osp.join(args.base_path, 'HeadHunter', 'test')
        seq_names = os.listdir(test_path)
        test_dataset = [osp.join(test_path, i, 'det', 'gt.txt') for i in seq_names]
        datasets = [HeadDataset(i,\
                            args.base_path,\
                            dataset_param={},\
                            train=False,\
                            name=j) for (i,j) in zip(test_dataset, seq_names) if os.stat(i).st_size > 0]
    else:
        datasets = [HeadDataset(args.test_dataset,
                                args.base_path,
                                dataset_param={},
                                train=False,
                                name=args.exp_name)]

    if args.n_gpu > 1:
        init_distributed_mode(args)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                      args.batch_size,
                                                      drop_last=False)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_sampler=batch_sampler,
                                                  num_workers=args.num_workers,
                                                  collate_fn=coco_collate)
        metric_logger = MetricLogger(delimiter="  ")
        header = 'Validation'
    else:
        model = model.cuda()
        data_loaders = [iter(data.DataLoader(i,\
                                           args.batch_size,\
                                           shuffle=False,\
                                           num_workers=args.num_workers,\
                                           collate_fn=coco_collate))\
                                        for i in datasets]
    
    eval_stats = defaultdict(list)
    eval_verbose = defaultdict(dict)
    
    for data_loader in tqdm(data_loaders):
        result_dict = evaluate(model, data_loader)
        print(result_dict)
        logging.info('Eval stats are {0}'.format(result_dict))
        for k,v in result_dict.items():
            eval_stats[k].append(v)
        eval_verbose[data_loader.dataset.name] = result_dict

    mean_eval_stat = {k:np.mean(v) for k,v in eval_stats.items()}
    print("Avg stats are ")
    print(mean_eval_stat)
    logging.info('Eval stats are {0}'.format(mean_eval_stat))
    print("Verbose eval results are ")
    print(eval_verbose)

if __name__ == '__main__':
    test()
