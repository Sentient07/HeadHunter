
import os.path as osp

import numpy as np
from scipy.misc import imread
from tqdm import tqdm

from .dataset import *
from .anchors import *
from .config import *

def compute_mean_std(test_dataset, base_path):
    """ Compute mean and std for dataset"""
    means = []
    stds = []
    with open(test_dataset, 'r') as infile:
        lines = infile.readlines()
        im_names = [i.rstrip().strip("#").lstrip() for i in lines if i.startswith('#')]
        im_path = [osp.join(base_path, i) for i in im_names]
        for imgs in tqdm(im_path):
            cur_im = imread(imgs)
            cur_std = np.std(cur_im, axis=(0, 1))
            cur_mean = np.mean(cur_im, axis=(0, 1))
            stds.append(cur_std)
            means.append(cur_mean)
    return np.mean(means, axis=0), np.mean(stds, axis=0)
