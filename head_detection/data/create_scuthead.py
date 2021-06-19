#!/usr/bin/env python
# coding: utf-8

# Creates scuthead training data

import numpy as np; import csv; from glob import glob; import os; import sys; 
from shutil import move; from tqdm import tqdm_notebook as tqdm; import os.path as osp  
import matplotlib.pyplot as plt; from PIL import Image 
import xml.etree.ElementTree as ET
import argparse

parser = argparse.ArgumentParser(description='Create training txt file for SCUT-HEAD')
parser.add_argument('--dset_path', help='Path to the root directory of the dataset')
parser.add_argument('--save_path', help='Places where files for detection are saved')
args = parser.parse_args()

np.random.seed(seed=12345)

def parse_xml(xml_files, prepend_path):
    det_list = []
    for sample_xml in xml_files:
        tree = ET.parse(sample_xml)
        root = tree.getroot()
        xml_fname = sample_xml.split('/')[-1].strip('.xml')

        fname = "#" + prepend_path + xml_fname + '.jpg'
        det_list.append(fname)
        obj_list = [child for child in root if child.tag=="object"]
        for objs in obj_list:
#             if objs.find('name').tail != 'head':
#                 import pdb;pdb.set_trace()
            startX, startY, endX, endY = [int(i.text) for i in objs.find('bndbox')]
            det_list.append([startX, startY, endX, endY, 1])
    return det_list

def create_scuthead_file(out_file, all_det_list, setfiles):
    set_ims = []
    for set_file in setfiles:
        with open(set_file) as f:
            set_ims.extend([line.rstrip() for line in f])
    with open(out_file, 'w+') as outfile:
        write_box = False
        for line in tqdm(all_det_list):
            if isinstance(line, str):
                fname = line.strip('#').lstrip().rstrip().split('/')[-1].split('.')[0]
                if fname not in set_ims:
                    write_box = False
                    continue
                write_box = True
                outfile.write("%s\n" % line.rstrip())
            else:
                if write_box:
                    assert isinstance(line, list)
                    str_list = " ".join(str(x) for x in line)
                    outfile.write("%s\n" % str_list)


def write_combined_det(all_det_list):
    ### Write the combined detection ###
    all_det_file = osp.join(args.out_path, 'AllDet.txt')
    with open(all_det_file, 'w') as f:
        for item in all_det_list:
            if isinstance(item, list):
                str_list = " ".join(str(x) for x in item)
                f.write("%s\n" % str_list)
            else:
                f.write("%s\n" % item)

if __name__ == '__main__':

    parta_xml_dir = osp.join(args.dset_path, 'ScutHead_A/Annotations/')
    partb_xml_dir = osp.join(args.dset_path, 'ScutHead_B/Annotations/')
    parta_prepend = "ScutHead/ScutHead_A/JPEGImages/"
    partb_prepend = "ScutHead/ScutHead_B/JPEGImages/"
    parta_det_list = parse_xml(sorted(glob(osp.join(parta_xml_dir, "*.xml"))), parta_prepend)
    partb_det_list = parse_xml(sorted(glob(osp.join(partb_xml_dir, "*.xml"))), partb_prepend)
    all_det_list = parta_det_list + partb_det_list

    # Create Trainval
    outfile_trainval = osp.join(args.out_path, 'SH_TrainVal.txt')
    setfile_trainval_a = osp.join(args.dset_path, 'ScutHead_A/ImageSets/Main/trainval.txt')
    setfile_trainval_b = osp.join(args.dset_path, 'ScutHead_B/ImageSets/Main/trainval.txt')
    create_scuthead_file(outfile_trainval, all_det_list,
                        [setfile_trainval_a, setfile_trainval_b])
    # Create Train
    outfile_train = osp.join(args.out_path, 'SH_Train.txt')
    setfile_train_a = osp.join(args.dset_path, 'ScutHead_A/ImageSets/Main/train.txt')
    setfile_train_b = osp.join(args.dset_path, 'ScutHead_B/ImageSets/Main/train.txt')
    create_scuthead_file(outfile_train, all_det_list,
                        [setfile_train_a, setfile_train_b])
    # Create val
    outfile_val = osp.join(args.out_path, '/SH_Val.txt')
    setfile_val_a = osp.join(args.dset_path, 'ScutHead_A/ImageSets/Main/val.txt')
    setfile_val_b = osp.join(args.dset_path, 'ScutHead_B/ImageSets/Main/val.txt')
    create_scuthead_file(outfile_val, all_det_list,
                        [setfile_val_a, setfile_val_b])
    # Create Test A
    outfile_test_parta = osp.join(args.out_path, 'SH_Test_PartA.txt')
    setfile_test_a = osp.join(args.dset_dir, 'ScutHead_A/ImageSets/Main/test.txt')
    create_scuthead_file(outfile_test_parta, all_det_list, [setfile_test_a])
    # Create Test B
    outfile_test_partb = osp.join(args.out_path, 'SH_Test_PartB.txt')
    setfile_test_b = osp.join(args.dset_dir, 'ScutHead_B/ImageSets/Main/test.txt')
    create_scuthead_file(outfile_test_partb, all_det_list, [setfile_test_b])

    # Create combined
    write_combined_det(all_det_list)