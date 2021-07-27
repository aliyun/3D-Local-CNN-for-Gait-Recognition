#! /usr/bin/env python
import os
import pdb
import time
import yaml
import json
import random
import shutil
import pickle
import argparse
import numpy as np
from tqdm import tqdm

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import AverageMeter
from solvers import Solver

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

__all__ = ['VisAnalysis']

def np2var(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).cuda()
    else:
        return x.cuda()

def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist

# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    view_num = acc.shape[0]
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / (view_num - 1)
    if not each_angle:
        result = np.mean(result)
    return result


class DataAnalysis(Solver):

    def frame_num_distribution(self):
        self.build_data()
        dataset = self.testloader.dataset

        num = len(dataset)
        frame_nums = np.array([len(os.listdir(i)) for i in dataset.seq_dir])

        self.print_log('Average # of frames: {}'.format(np.mean(frame_nums)))

        sns.distplot(frame_nums, hist=True, kde=True)
        save_path = os.path.join(self.cfg.work_dir, self.cfg.save_name + '.pdf')
        plt.savefig(save_path)
        print('Saving to {}'.format(save_path))
        plt.close()

