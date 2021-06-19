#!/usr/bin/env python
# coding: utf-8

import os.path as osp
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.utils.data as data
from scipy.misc import imsave
from torchvision import transforms
from tqdm import tqdm
from albumentations.pytorch import ToTensor
from head_detection.vision.utils import collate_fn as coco_collate


def to_torch(im):
    transf = ToTensor()
    torched_im = transf(image=im)['image'].to(torch.device("cuda"))
    return torch.unsqueeze(torched_im, 0)


def get_state_dict(net, pt_model, only_backbone=False):
    """
    Restore weight. Full or partial depending on `only_backbone`.
    """
    strict = False if only_backbone else True
    state_dict = torch.load(pt_model)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        # Remove Head module while restoring network
        if only_backbone:
            if 'Head' in name.split('.')[0]:
                continue
            else:
                name = 'backbone.' + name
        new_state_dict[name] = v
    return new_state_dict


def restore_network(net, pt_model, only_backbone=False):
    """
    Restore weight. Full or partial depending on `only_backbone`.
    """
    strict = False if only_backbone else True
    state_dict = torch.load(pt_model)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        # Remove Head module while restoring network
        if only_backbone:
            if 'Head' in name.split('.')[0]:
                continue
            else:
                name = 'backbone.' + name
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict, strict=strict)
    print('Loaded the entire model in %r mode' %strict)
    return net


def visualize(base_path, test_dataset, plot_dir, batch_size=4, ):
    """Visualize ground truth data"""
    device = torch.device('cuda')
    dataset = HeadDataset(test_dataset,
                          base_path,
                          dataset_param={},
                          train=False)
    batch_iterator = iter(data.DataLoader(dataset, batch_size,
                                          shuffle=False,
                                          num_workers=4,
                                          collate_fn=coco_collate))
    for ind, (images, targets) in enumerate(tqdm(batch_iterator)):
        images = list(img.to(device) for img in images)
        np_images = [(ims.cpu().numpy()*255.).astype(np.uint8) for ims in images]
        gt_boxes = [gt['boxes'].numpy().astype(np.float64) for gt in targets]
        for np_im, gt_box in zip(np_images, gt_boxes):
            plot_images = plot_ims(np_im, [], gt_box)
            imsave(osp.join(plot_dir, str(ind) + '.jpg'), plot_images)


def plot_ims(img, pred_box, gt_box=None, text=True):
    """
    Prediction : Yellow
    Ground Truth : Green
    """
    plotting_im = img.transpose(1,2,0).copy()
    if gt_box is None:
        gt_box = []

    for b_id, box in enumerate(gt_box):
        (startX, startY, endX, endY) = [int(i) for i in box]
        cv2.rectangle(plotting_im, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        cur_centroid = tuple([(startX+endX)//2,
                              (startY+endY)//2])
        if text:
            cv2.putText(plotting_im, str(b_id), cur_centroid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for b_id, box in enumerate(pred_box):
        (startX, startY, endX, endY) = [int(i) for i in box]
        cv2.rectangle(plotting_im, (startX, startY), (endX, endY),
                      (255, 255, 0), 2)
        cur_centroid = tuple([(startX+endX)//2,
                              (startY+endY)//2])
        if text:
            cv2.putText(plotting_im, str(b_id), cur_centroid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    return plotting_im
