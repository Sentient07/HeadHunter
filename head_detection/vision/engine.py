#!/usr/bin/env python
# coding: utf-8

import math
import sys
import time
from collections import defaultdict

import brambox
import pandas as pd
import numpy as np
import torch
import torchvision.models.detection.mask_rcnn

from head_detection.vision import utils
from brambox.stat._matchboxes import match_det, match_anno
from brambox.stat import coordinates, mr_fppi, ap, pr, threshold, fscore, peak, lamr


def check_empty_target(targets):
    for tar in targets:
        if len(tar['boxes']) < 1:
            return True
    return False


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        if check_empty_target(targets):
            continue
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def safe_div(x,y):
    if y == 0:
        return 0
    return x / y

def get_moda(det, anno, threshold=0.2, ignore=None):
    if ignore is None:
        ignore = anno.ignore.any()

    dets_per_frame = anno.groupby('image').filter(lambda x: any(x['ignore'] == 0))
    dets_per_frame = dets_per_frame.groupby('image').size().to_dict()
    # Other param for finding matched anno
    crit = coordinates.pdollar if ignore else coordinates.iou
    label = len({*det.class_label.unique(), *anno.class_label.unique()}) > 1
    matched_dets = match_det(det, anno, threshold, criteria=crit,
                            class_label=label, ignore=2 if ignore else 0)
    fp_per_im = matched_dets[matched_dets.fp==True].groupby('image').size().to_dict()
    tp_per_im = matched_dets[matched_dets.tp==True].groupby('image').size().to_dict()
    valid_anno = anno[anno.ignore == False].groupby('image').size().to_dict()
    assert valid_anno.keys() == tp_per_im.keys()

    moda_ = []
    for k, _ in valid_anno.items():
        n_gt = valid_anno[k]
        miss = n_gt-tp_per_im[k]
        fp = fp_per_im[k]
        moda_.append(safe_div((miss+fp), n_gt))
    return 1 - np.mean(moda_)


def get_modp(det, anno, threshold=0.2, ignore=None):
    if ignore is None:
        ignore = anno.ignore.any()
    # Compute TP/FP
    if not {'tp', 'fp'}.issubset(det.columns):
        crit = coordinates.pdollar if ignore else coordinates.iou
        label = len({*det.class_label.unique(), *anno.class_label.unique()}) > 1
        det = match_anno(det, anno, threshold, criteria=crit, class_label=label, ignore=2 if ignore else 0)
    elif not det.confidence.is_monotonic_decreasing:
        det = det.sort_values('confidence', ascending=False)
    modp = det.groupby('image')['criteria'].mean().mean()
    return modp

@torch.no_grad()
def evaluate(model, data_loader, out_path=None, benchmark=None):
    """
    Evaluates a model over testing set, using AP, Log MMR, F1-score
    """
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    device=torch.device('cuda')
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Valid:'

    # Brambox eval related
    pred_dict = defaultdict(list)
    gt_dict = defaultdict(list)
    results = {}
    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        evaluator_time = time.time()
        # Pred lists
        pred_boxes = [p['boxes'].numpy() for p in outputs]
        pred_scores = [p['scores'].numpy() for p in outputs]

        # GT List
        gt_boxes = [gt['boxes'].numpy()for gt in targets]

        # ignore variables are used in our benchmark and CHuman Benchmark
        ignore_ar = [gt['ignore'] for gt in targets]
        # Just to be sure target and prediction have batchsize 2
        assert len(gt_boxes) == len(pred_boxes)
        for j in range(len(gt_boxes)):
            im_name = str(targets[j]['image_id']) + '.jpg'
            # write to results dict for MOT format
            results[targets[j]['image_id'].item()] = {'boxes': pred_boxes[j],
                                                      'scores': pred_scores[j]}
            for _, (p_b, p_s) in enumerate(zip(pred_boxes[j], pred_scores[j])):
                pred_dict['image'].append(im_name)
                pred_dict['class_label'].append('head')
                pred_dict['id'].append(0)
                pred_dict['x_top_left'].append(p_b[0])
                pred_dict['y_top_left'].append(p_b[1])
                pred_dict['width'].append(p_b[2] - p_b[0])
                pred_dict['height'].append(p_b[3] - p_b[1])
                pred_dict['confidence'].append(p_s)

            for _, (gt_b, ignore_val) in enumerate(zip(gt_boxes[j], ignore_ar[j])):
                gt_dict['image'].append(im_name)
                gt_dict['class_label'].append('head')
                gt_dict['id'].append(0)
                gt_dict['x_top_left'].append(gt_b[0])
                gt_dict['y_top_left'].append(gt_b[1])
                gt_dict['width'].append(gt_b[2] - gt_b[0])
                gt_dict['height'].append(gt_b[3] - gt_b[1])
                gt_dict['ignore'].append(ignore_val)

        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
    
    # Save results in MOT format if out_path is provided
    if out_path is not None:
        data_loader.dataset.write_results_files(results, out_path)
    # gather the stats from all processes
    pred_df = pd.DataFrame(pred_dict)
    gt_df = pd.DataFrame(gt_dict)
    pred_df['image'] = pred_df['image'].astype('category')
    gt_df['image'] = gt_df['image'].astype('category')
    pr_ = pr(pred_df, gt_df,  ignore=True)
    ap_ = ap(pr_)
    mr_fppi_ = mr_fppi(pred_df, gt_df, threshold=0.5,  ignore=True)
    lamr_ = lamr(mr_fppi_)
    f1_ = fscore(pr_)
    f1_ = f1_.fillna(0)
    threshold_ = peak(f1_)

    moda = get_moda(pred_df, gt_df, threshold=0.2, ignore=True)
    modp = get_modp(pred_df, gt_df, threshold=0.2, ignore=True)

    result_dict = {'AP' : ap_, 'MMR' : lamr_,
                    'f1' : threshold_.f1, 'r':pr_['recall'].values[-1],
                    'moda' : moda, 'modp' : modp}

    metric_logger.synchronize_between_processes()

    torch.set_num_threads(n_threads)
    return result_dict
