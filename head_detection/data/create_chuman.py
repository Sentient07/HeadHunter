#!/usr/bin/env python
# coding: utf-8

"""
##############################
### CrowdHuman Processing ###
#############################
"""
import json
import argparse

import os.path as osp
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Create training txt file for SCUT-HEAD')
parser.add_argument('--base_dir', help='Path to the root directory of the dataset')
parser.add_argument('--save_prepend', help='Places where files for detection are saved')
args = parser.parse_args()

def parse_odgt(base_dir, save_prepend, fname):
    out_list = []
    chuman_wh_list = []

    with open(osp.join(base_dir, fname), 'r') as odt_f:
        for images in tqdm(odt_f.readlines()):
            s_det = json.loads(images)
            cur_imname = s_det.get('ID') + '.jpg'
            fname = "#" + save_prepend + cur_imname
            out_list.append(fname)
#             cur_boxes = [gt_box['hbox'] for gt_box in s_det['gtboxes'] 
#                             if bool(gt_box.get('head_attr', None))]
            for gt_box in s_det['gtboxes']:
                ignore_label = 0
                is_mask = gt_box.get('tag')
                head_attr = gt_box.get('head_attr', None)
                if not bool(head_attr):
                    continue
                if is_mask == 'mask':
                    ignore_label = -1
                ignore_cond = head_attr.get('ignore') == 1
                if ignore_cond:
                    ignore_label = -1
                (startX, startY, W, H) = gt_box['hbox']
                chuman_wh_list.append([W, H])
                endX, endY = startX+W, startY+H
                new_coord = [startX, startY, endX, endY, ignore_label]
                new_line = " ".join(str(x) for x in new_coord)
                out_list.append(new_line)
                
    return out_list

def write_out(out_file, lines):
    with open(out_file, 'w+') as of:
        for new_line in tqdm(lines):
            of.write("%s\n" % new_line)

if __name__ == '__main__':
    im_dir = osp.join(args.base_dir, 'Images')
    train_out = osp.join(args.base_dir, 'CHuman_Train.txt')
    val_out = osp.join(args.base_dir, 'CHuman_Valid.txt')
    train_lines = parse_odgt(args.base_dir, args.save_prepend, 'annotation_train.odgt')
    val_lines = parse_odgt(args.base_dir, args.save_prepend, 'annotation_val.odgt')
    write_out(train_out, train_lines)
    write_out(val_out, val_lines)