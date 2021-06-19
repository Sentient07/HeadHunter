#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import os.path as osp

import cv2
import torch
import yaml

from head_detection.data import (HeadDataset, cfg_mnet, cfg_res50,
                                 cfg_res50_4fpn, cfg_res152, ch_anchors,
                                 combined_anchors, compute_mean_std,
                                 headhunt_anchors, sh_anchors)
from head_detection.models.head_detect import customRCNN
from head_detection.utils import restore_network
from head_detection.vision.engine import evaluate, train_one_epoch
from head_detection.vision.utils import collate_fn, init_distributed_mode

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


parser = argparse.ArgumentParser(description='Training of Head detector')
# Dataset related arguments
parser.add_argument('--cfg_file', required=True,
                    help='Config file')
# Torch DataParallel args
parser.add_argument('--world_size', default=1,
                    type=int, help='number of distributed processes')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
args = parser.parse_args()
print(args)
# Get variables from config file
with open(args.cfg_file, 'r') as stream:
    CONFIG = yaml.safe_load(stream)
print(CONFIG)
DATASET_CFG = CONFIG['DATASET']
TRAIN_CFG = CONFIG['TRAINING']
HYP_CFG = CONFIG['HYPER_PARAM']
NET_CFG = CONFIG['NETWORK']


# Create Logging files to log validation losses
log_name = osp.join(TRAIN_CFG['log_dir'],
                    TRAIN_CFG['exp_name'] + ".log")
logging.basicConfig(filename=log_name, filemode='a', level=logging.INFO)
logging.info("Writing logs to this file" + str(log_name))
print("Logging into %s" %log_name)

# anchors -> use mean
benchmark = DATASET_CFG['benchmark']
cfg = cfg_res50_4fpn
# Set the device
if torch.cuda.is_available():
    device = torch.device('cuda')


def train():
    """ Train the Head detector """
    init_distributed_mode(args)
    save_dir = TRAIN_CFG['save_dir']
    if not os.path.exists(save_dir) and torch.distributed.get_rank() == 0:
        os.mkdir(save_dir)
    kwargs = {}
    # If augmenting data, disable Pytorch's own augmentataion
    # This has to be done manually as augmentation is embedded
    # refer : https://github.com/pytorch/vision/issues/2263
    base_path = DATASET_CFG['base_path']
    train_set = DATASET_CFG['train']
    valid_set = DATASET_CFG['valid']
    dset_mean_std = DATASET_CFG['mean_std']
    if dset_mean_std is not None:
        dataset_mean = [i/255. for i in dset_mean_std[0]]
        dataset_std = [i/255. for i in dset_mean_std[1]]
    else:
        dataset_mean, dataset_std = compute_mean_std(base_path, train_set)
    kwargs['image_mean'] = dataset_mean
    kwargs['image_std'] = dataset_std
    kwargs['min_size'] = DATASET_CFG['min_size']
    kwargs['max_size'] = DATASET_CFG['max_size']
    kwargs['box_detections_per_img'] = 300 # increase max det to max val in our benchmark

    # Set benchmark related parameters
    if benchmark == 'ScutHead':
        combined_cfg = {**cfg, **sh_anchors}
    elif benchmark == 'CrowdHuman':
        combined_cfg = {**cfg, **ch_anchors}
    elif benchmark == 'Combined':
        combined_cfg = {**cfg, **combined_anchors}
    else:
        raise ValueError("New dataset has to be registered")

    # Create Model
    default_filter = False
    model = customRCNN(cfg=combined_cfg,
                        use_deform=NET_CFG['use_deform'],
                        ohem=NET_CFG['ohem'],
                        context=NET_CFG['context'],
                        custom_sampling=NET_CFG['custom_sampling'],
                        default_filter=default_filter,
                        soft_nms=NET_CFG['soft_nms'],
                        upscale_rpn=NET_CFG['upscale_rpn'],
                        median_anchors=NET_CFG['median_anchors'],
                        **kwargs).cuda()    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=True)
        model_without_ddp = model.module

    # Create Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=HYP_CFG['learning_rate'],
                                momentum=HYP_CFG['learning_rate'],
                                weight_decay=HYP_CFG['weight_decay'])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=TRAIN_CFG['milestones'],
                                                    gamma=HYP_CFG['gamma'])
    # Restore from checkpoint
    pt_model = TRAIN_CFG['pretrained_model']
    if pt_model:
        model_without_ddp = restore_network(model_without_ddp, pt_model,
                                            only_backbone=TRAIN_CFG['only_backbone'])
    
    # Create training and vaid dataset
    dataset_param = {'mean': dataset_mean, 'std':dataset_std,
                    'shape':(kwargs['min_size'], kwargs['max_size'])}
    batch_size = HYP_CFG['batch_size']
    train_dataset = HeadDataset(train_set,
                                base_path,
                                dataset_param,
                                train=True)
    val_dataset = HeadDataset(valid_set,
                              base_path,
                              dataset_param,
                              train=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler,
                                                        batch_size,
                                                        drop_last=True)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_sampler=train_batch_sampler,
                                                    num_workers=args.num_workers,
                                                    collate_fn=collate_fn)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_batch_sampler = torch.utils.data.BatchSampler(val_sampler,
                                                      batch_size,
                                                      drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_sampler=val_batch_sampler,
                                                  num_workers=args.num_workers,
                                                  collate_fn=collate_fn)
    # Fastforward the LR decayer
    start_epoch = TRAIN_CFG['start_epoch']
    max_epoch = TRAIN_CFG['max_epoch']
    for _ in range(0, -1):
        scheduler.step()

    # Start training
    print("======= Training for " + str(max_epoch) + "===========")
    for epoch in range(start_epoch, int(max_epoch) + 1):
        if epoch % TRAIN_CFG['eval_every'] == 0:
            print("========= Evaluating Model ==========")
            result_dict = evaluate(model, val_data_loader, benchmark=benchmark)
            if torch.distributed.get_rank() == 0:
                logging.info('Eval score at {0} epoch is {1}'.format(str(epoch),
                            result_dict))
        
        train_one_epoch(model, optimizer, train_data_loader,
                        device, epoch, print_freq=1000)
        scheduler.step()
        if torch.distributed.get_rank() == 0:
            print("Saving model")
            torch.save(model.state_dict(), osp.join(save_dir,
                       TRAIN_CFG['exp_name'] + '_epoch_' + str(epoch) + '.pth'))

if __name__ == '__main__':
    train()
