#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : ZhenHuang
Date               : 2021-08-13 09:54
Last Modified By   : ZhenHuang
Last Modified Date : 2021-08-16 16:23
Description        : Generate Gait GIFs
-------- 
Copyright (c) 2021 Alibaba Inc. 
'''

#! /usr/bin/env python
import os
import cv2
import math
import random
import pickle
import imageio
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import AverageMeter
from solvers import Solver

from sklearn.manifold import TSNE
# import umap
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

__all__ = ['Visualization']

from .gif_cfg import sampling, cfg

def np2var(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).cuda()
    else:
        return x.cuda()


def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(
        x**2, 1).unsqueeze(1) + torch.sum(y**2, 1).unsqueeze(1).transpose(
            0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    view_num = acc.shape[0]
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / (view_num - 1)
    if not each_angle:
        result = np.mean(result)
    return result


class Visualization(Solver):
    def generate_sampled_gifs_v1(self):
        self.build_data()
        dataset = self.testloader.dataset
        N = len(dataset)
        a = random.randint(0, N - 1)
        a = 0
        print('{}/{}'.format(a, N))
        frames = dataset[a][0] * 255
        frames = frames.astype('uint8')
        frames = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in frames]
        imageio.mimsave('./seq.gif', frames, 'GIF', duration=0.1)

        bboxes = {
            'head': (1, 9, 14, 30),  # (h-top, h-bottom, w-left, w-right)
            'torso': [8, 40, 2, 42],
            'armL': [26, 38, 7, 19],
            'armR': [26, 38, 25, 37],
            'legL': [39, 63, 3, 23],
            'legR': [39, 63, 21, 41],
        }

        def process(img, bbox):
            H, W, C = img.shape
            out = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            out_h = int(1. * W * out.shape[0] / out.shape[1])
            out = cv2.resize(out, (W, out_h), interpolation=cv2.INTER_CUBIC)
            pad_t = (H - out_h) // 2
            pad_b = H - pad_t - out_h
            out = np.pad(out, ((pad_t, pad_b), (0, 0), (0, 0)))
            return out

        imageio.mimsave('./head.gif',
                        [process(f, bboxes['head']) for f in frames],
                        'GIF',
                        duration=0.1)
        imageio.mimsave('./torso.gif',
                        [process(f, bboxes['torso']) for f in frames],
                        'GIF',
                        duration=0.1)
        imageio.mimsave('./armL.gif',
                        [process(f, bboxes['armL']) for f in frames],
                        'GIF',
                        duration=0.1)
        imageio.mimsave('./armR.gif',
                        [process(f, bboxes['armR']) for f in frames],
                        'GIF',
                        duration=0.1)
        imageio.mimsave('./legL.gif',
                        [process(f, bboxes['legL']) for f in frames],
                        'GIF',
                        duration=0.1)
        imageio.mimsave('./legR.gif',
                        [process(f, bboxes['legR']) for f in frames],
                        'GIF',
                        duration=0.1)

    def generate_sampled_gifs_v2(self):
        self.build_data()
        dataset = self.testloader.dataset
        N = len(dataset)
        # for i in range(N):
        #     frames, view, seq_type, label = dataset[i]
        #     if view == '126' and seq_type[:2] == 'nm':
        #         print(i, len(frames))
                # return
        a = 530
        frames = dataset[a][0] * 255
        frames = frames.astype('uint8')
        frames = [cv2.cvtColor(f, cv2.COLOR_GRAY2RGB) for f in frames]
        print('{}/{}, {} frames'.format(a, N, len(frames)))

        imageio.mimsave('./seq.gif', frames, 'GIF', duration=0.1)

        T = len(frames)
        head, torso, armL, armR, legL, legR = cfg(a)
        bbox = {
            'head': head,
            'torso': torso,
            'armL': armL,
            'armR': armR,
            'legL': legL,
            'legR': legR
        }

        for k, v in bbox.items():
            imageio.mimsave('./{}.gif'.format(k),
                [sampling(f, b) for f, b in zip(frames, v)],
                'GIF',
                duration=0.1)


    def generate_sampled_sequences(self):
        work_dir = self.cfg.work_dir

        self.build_data()
        dataset = self.testloader.dataset
        N = len(dataset)
        # for i in range(N):
        #     frames, view, seq_type, label = dataset[i]
        #     if view == '126' and seq_type[:2] == 'nm':
        #         print(i, len(frames))
                # return
        a = 49
        frames = dataset[a][0] * 255
        frames = frames.astype('uint8')
        frames = [cv2.cvtColor(f, cv2.COLOR_GRAY2RGB) for f in frames]
        print('{}/{}, {} frames'.format(a, N, len(frames)))

        imageio.mimsave('./seq.gif', frames, 'GIF', duration=0.1)
        save_dir = os.path.join(work_dir, 'original')
        os.makedirs(save_dir, exist_ok=True)
        for i, im in enumerate(frames):
            save_path = os.path.join(save_dir, '{}.jpg'.format(i))
            imageio.imsave(save_path, im)

        T = len(frames)
        head, torso, armL, armR, legL, legR = cfg(a)
        bbox = {
            'head': head,
            'torso': torso,
            'armL': armL,
            'armR': armR,
            'legL': legL,
            'legR': legR
        }

        for k, v in bbox.items():
            save_dir = os.path.join(work_dir, k)
            os.makedirs(save_dir, exist_ok=True)
            seq = [sampling(f, b) for f, b in zip(frames, v)]
            for i, im in enumerate(seq):
                save_path = os.path.join(save_dir, '{}.jpg'.format(i))
                imageio.imsave(save_path, im)


    def generate_gifs(self):
        self.build_data()
        dataset = self.testloader.dataset
        N = len(dataset)
        print('N: ', N)
        a = random.randint(0, N - 1)
        a = 0
        frames = dataset[a][0] * 255
        frames = frames.astype('uint8')
        frames = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in frames]
        imageio.mimsave('./seq.gif', frames, 'GIF', duration=0.1)

        colors = {
            'head': (239, 138, 55),
            'torso': (91, 196, 110),
            'armL': (252, 240, 82),
            'armR': (241, 127, 132),
            'legL': (178, 232, 252),
            'armR': (244, 159, 251),
        }
        bbox = {
            'head': [1, 9, 14, 30],  # (h-top, h-bottom, w-left, w-right)
            'torso': [],
            'armL': [],
            'armR': [],
            'legL': [],
            'legR': [],
        }
        head = self.generate_part_gif(frames, colors['head'], bbox['head'])
        imageio.mimsave('./head.gif', head, 'GIF', duration=0.1)

    def draw_mask(self, frames, color=None, bbox=None):
        out = []
        for f in frames:
            zero_mask = np.zeros((f.shape), dtype=np.uint8)
            zero_mask = cv2.rectangle(zero_mask, (bbox[2], bbox[0]),
                                      (bbox[3], bbox[1]),
                                      color=color,
                                      thickness=-1)
            zero_mask = np.array(zero_mask)
            out.append(cv2.addWeighted(f, 1, zero_mask, 0.5, 0))
        return out
