#!/usr/bin/env python
# coding: utf-8

from collections import OrderedDict

import torch
import torchvision.models as models
from torchvision.models.detection.rpn import AnchorGenerator

from head_detection.models.fast_rcnn import FasterRCNN
from head_detection.models.net import BackBoneWithFPN
from head_detection.models.net import MobileNetV1 as MobileNetV1


def create_backbone(cfg, use_deform=False,
                    context=None, default_filter=False):
    """Creates backbone """
    in_channels = cfg['in_channel']    
    if cfg['name'] == 'Resnet50':
        feat_ext = models.resnet50(pretrained=cfg['pretrain'])
        if len(cfg['return_layers']) == 3:
            in_channels_list = [
                in_channels * 2,
                in_channels * 4,
                in_channels * 8,
            ]
        elif len(cfg['return_layers']) == 4:
            in_channels_list = [
                    in_channels,
                    in_channels * 2,
                    in_channels * 4,
                    in_channels * 8,
            ]
        else:
            raise ValueError("Not yet ready for 5FPN")
    elif cfg['name'] == 'Resnet152':
        feat_ext = models.resnet152(pretrained=cfg['pretrain'])
        in_channels_list = [
            in_channels,
            in_channels * 2,
            in_channels * 4,
            in_channels * 8,
        ]
    elif cfg['name'] == 'mobilenet0.25':
        feat_ext = MobileNetV1()
        in_channels_list = [
            in_channels * 2,
            in_channels * 4,
            in_channels * 8,
        ]
        if cfg['pretrain']:
            checkpoint = torch.load("./Weights/mobilenetV1X0.25_pretrain.tar",
                                    map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:]  # remove module.
                new_state_dict[name] = v
                # load params
            feat_ext.load_state_dict(new_state_dict)
    else:
        raise ValueError("Unsupported backbone")

    out_channels = cfg['out_channel']
    backbone_with_fpn = BackBoneWithFPN(feat_ext, cfg['return_layers'],
                                        in_channels_list,
                                        out_channels,
                                        context_module=context,
                                        use_deform=use_deform,
                                        default_filter=default_filter)
    return backbone_with_fpn


def customRCNN(cfg, use_deform=False,
              ohem=False, context=None, custom_sampling=False,
              default_filter=False, soft_nms=False,
              upscale_rpn=False, median_anchors=True,
              **kwargs):
    
    """
    Calls a Faster-RCNN head with custom arguments + our backbone
    """

    backbone_with_fpn = create_backbone(cfg=cfg, use_deform=use_deform,
                                        context=context,
                                        default_filter=default_filter)
    if median_anchors:
        anchor_sizes = cfg['anchor_sizes']
        aspect_ratios = cfg['aspect_ratios']
        rpn_anchor_generator = AnchorGenerator(anchor_sizes,
                                           aspect_ratios)
        kwargs['rpn_anchor_generator'] = rpn_anchor_generator

    if custom_sampling:
        # Random hand thresholding ablation experiment to understand difference
        # in behaviour of Body vs head bounding boxes
        kwargs['rpn_fg_iou_thresh'] = 0.5
        kwargs['box_bg_iou_thresh'] = 0.4
        kwargs['box_positive_fraction'] = 0.5
        # kwargs['box_nms_thresh'] = 0.7

    kwargs['cfg'] = cfg
    model = FasterRCNN(backbone_with_fpn, num_classes=2, ohem=ohem, soft_nms=soft_nms,
                       upscale_rpn=upscale_rpn, **kwargs)
    return model
