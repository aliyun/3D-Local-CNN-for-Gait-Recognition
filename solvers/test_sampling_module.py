#! /usr/bin/env python
import os
import pdb
import time
import yaml
import json
import random
import shutil
import argparse
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from utils import AverageMeter, import_class, LearningRate, init_seed
from solvers import Solver



class TestSamplingModule(Solver):
    def visualize_sequence(self, seq, name, index=None):
        seq = seq.squeeze().detach().cpu().numpy()
        if index is not None:
            assert len(seq) == len(index), '{} vs. {}'.format(len(seq), len(index))
        else:
            index = np.arange(len(seq))
        for i in range(len(seq)):
            sub_name = os.path.join(name, str(index[i]))
            # tensorboard
            self.writer.add_image(sub_name, seq[i], dataformats='HW')
            # matplot
            # fig = plt.figure()
            # plt.imshow(seq[i], plt.cm.gray)
            # save_path = os.path.join(self.work_dir, sub_name + '.pdf')
            # plt.savefig(save_path)
            # plt.close()
            # self.print_log('Save figure to {}'.format(save_path))


    def train(self):
        self.build_data()
        num_seqs = len(self.testloader.dataset)
        seq_index = np.random.randint(0, num_seqs, (1,))[0]
        seq, _, _, _ = self.testloader.dataset[seq_index]
        seq = torch.from_numpy(seq).float().squeeze().cuda()

        print(seq.shape)
        self.visualize_sequence(seq, 'original')

        # head
        sampled, index = self.fix_sampling(seq, 64, 44, 8, 16,
                                           0.5, 0.5, 1.0/16,
                                           0.3, 0.1, 0.4, 4.0/11)
        self.visualize_sequence(sampled, 'head', index)
        sampled, index = self.fix_sampling(seq, 64, 44, 8, 16,
                                           0.5, 0.5, 1.0/16,
                                           0.3, 0.1, 0.4, 4.0/11,
                                           inverse=True)
        self.visualize_sequence(sampled, 'head_inv', index)

        # torso
        sampled, index = self.fix_sampling(seq, 64, 44, 32, 40,
                                           0.5, 0.5, 3.0/8,
                                           0.3, 0.1, 0.4, 10.0/11)
        self.visualize_sequence(sampled, 'torso', index)
        sampled, index = self.fix_sampling(seq, 64, 44, 32, 40,
                                           0.5, 0.5, 3.0/8,
                                           0.3, 0.1, 0.4, 10.0/11,
                                           inverse=True)
        self.visualize_sequence(sampled, 'torso_inv', index)

        # left arm
        sampled, index = self.fix_sampling(seq, 64, 44, 12, 12,
                                           0.5, 0.3, 0.5,
                                           0.3, 0.1, 0.4, 3.0/11)
        self.visualize_sequence(sampled, 'Larm', index)
        sampled, index = self.fix_sampling(seq, 64, 44, 12, 12,
                                           0.5, 0.3, 0.5,
                                           0.3, 0.1, 0.4, 3.0/11,
                                           inverse=True)
        self.visualize_sequence(sampled, 'Larm_inv', index)
        # right arm
        sampled, index = self.fix_sampling(seq, 64, 44, 12, 12,
                                           0.5, 0.7, 0.5,
                                           0.3, 0.1, 0.4, 3.0/11)
        self.visualize_sequence(sampled, 'Rarm', index)
        sampled, index = self.fix_sampling(seq, 64, 44, 12, 12,
                                           0.5, 0.7, 0.5,
                                           0.3, 0.1, 0.4, 3.0/11,
                                           inverse=True)
        self.visualize_sequence(sampled, 'Rarm_inv', index)

        # left leg
        sampled, index = self.fix_sampling(seq, 64, 44, 24, 20,
                                           0.5, 0.3, 0.8,
                                           0.3, 0.1, 0.4, 5.0/11)
        self.visualize_sequence(sampled, 'Lleg', index)
        sampled, index = self.fix_sampling(seq, 64, 44, 24, 20,
                                           0.5, 0.3, 0.8,
                                           0.3, 0.1, 0.4, 5.0/11,
                                           inverse=True)
        self.visualize_sequence(sampled, 'Lleg_inv', index)
        sampled, index = self.fix_sampling(seq, 64, 44, 24, 20,
                                           0.5, 0.7, 0.8,
                                           0.3, 0.1, 0.4, 5.0/11)
        self.visualize_sequence(sampled, 'Rleg', index)
        sampled, index = self.fix_sampling(seq, 64, 44, 24, 20,
                                           0.5, 0.7, 0.8,
                                           0.3, 0.1, 0.4, 5.0/11,
                                           inverse=True)
        self.visualize_sequence(sampled, 'Rleg_inv', index)

    def fix_sampling(self, seq, out_h, out_w, in_h, in_w,
                     dt_offset=0.5, dx_offset=0.5, dy_offset=1.0/16,
                     sigma_t_offset=0.3, sigma_offset=0.1,
                     delta_t_offset=0.4, delta_offset=4.0/11,
                     inverse=False, eps=1e-8):
        T, H, W = seq.size()
        atten_out_t = T
        atten_out_w = out_w
        atten_out_h = int(round(atten_out_w / in_w * in_h))
        anchor_t = T * dt_offset
        anchor_x = W * dx_offset
        anchor_y = H * dy_offset

        """ get localization parameters """
        dx = anchor_x
        dy = anchor_y
        dt = anchor_t
        sigma2_t = sigma_t_offset
        sigma2 = sigma_offset
        delta_t = delta_t_offset
        delta = delta_offset

        """ set up transform matrix """
        grid_t_i = torch.arange(0, atten_out_t).view(1, -1).float().cuda().detach()
        grid_x_i = torch.arange(0, atten_out_w).view(1, -1).float().cuda().detach()
        grid_y_i = torch.arange(0, atten_out_h).view(1, -1).float().cuda().detach()
        mu_t = dt + (grid_t_i - atten_out_t / 2.0) * delta_t
        mu_x = dx + (grid_x_i - atten_out_w / 2.0) * delta
        mu_y = dy + (grid_y_i - atten_out_h / 2.0) * delta

        c = torch.arange(0, T).view(1,-1).float().cuda().detach()
        a = torch.arange(0, W).view(1,-1).float().cuda().detach()
        b = torch.arange(0, H).view(1,-1).float().cuda().detach()
        mu_t = mu_t.view(atten_out_t, 1)
        mu_x = mu_x.view(atten_out_w, 1)
        mu_y = mu_y.view(atten_out_h, 1)
        transform_t = torch.exp(-1 * torch.pow(c - mu_t, 2) / (2*sigma2_t))
        transform_x = torch.exp(-1 * torch.pow(a - mu_x, 2) / (2*sigma2))
        transform_y = torch.exp(-1 * torch.pow(b - mu_y, 2) / (2*sigma2))
        # normalize, sum over H and W dims
        eps_tensor_t = eps * torch.ones(T).cuda().detach()
        eps_tensor_h = eps * torch.ones(H).cuda().detach()
        eps_tensor_w = eps * torch.ones(W).cuda().detach()
        # TODO inverse transform
        Ft = transform_t / torch.max(torch.sum(transform_t, 1, keepdim=True), eps_tensor_t)
        Fx = transform_x / torch.max(torch.sum(transform_x, 1, keepdim=True), eps_tensor_w)
        Fy = transform_y / torch.max(torch.sum(transform_y, 1, keepdim=True), eps_tensor_h)

        Ftv = Ft.view(Ft.size(0), Ft.size(1))
        Fyv = Fy.view(1, Fy.size(0), Fy.size(1))
        Fxv = Fx.view(1, Fx.size(0), Fx.size(1))
        Fxt = torch.transpose(Fxv, 1, 2)
        glimpse = torch.matmul(Fyv, torch.matmul(seq, Fxt))
        glimpse = glimpse.view(glimpse.size(0), -1)
        glimpse = torch.matmul(Ftv, glimpse)
        glimpse = glimpse.view(atten_out_t, atten_out_h,
                               atten_out_w)
        if inverse == True:
            inv_Ft = transform_t / torch.max(torch.sum(transform_t, 0, keepdim=True), eps_tensor_t)
            inv_Fx = transform_x / torch.max(torch.sum(transform_x, 0, keepdim=True), eps_tensor_w)
            inv_Fy = transform_y / torch.max(torch.sum(transform_y, 0, keepdim=True), eps_tensor_h)
            inv_Ftv = inv_Ft.view(inv_Ft.size(0), inv_Ft.size(1))
            inv_Fyv = inv_Fy.view(1, inv_Fy.size(0), inv_Fy.size(1))
            inv_Fxv = inv_Fx.view(1, inv_Fx.size(0), inv_Fx.size(1))
            inv_Fyt = torch.transpose(inv_Fyv, 1, 2)
            glimpse = torch.matmul(inv_Fyt, torch.matmul(glimpse, inv_Fxv))
            inv_Ftt = torch.transpose(inv_Ftv, 0, 1)
            glimpse = glimpse.view(glimpse.size(0), -1)
            glimpse = torch.matmul(inv_Ftt, glimpse)
            glimpse = glimpse.view(T, H, W)

        x_h = glimpse.size(1)
        pad_t = (out_h - x_h) // 2
        pad_b = out_h - pad_t - x_h
        out = F.pad(glimpse, pad=(0, 0, pad_t, pad_b))

        if inverse == True:
            index = np.arange(len(seq))
        else:
            index = mu_t.squeeze().cpu().numpy()
        return out, index


    def test_original_samplers(self):
        self.build_data()
        num_seqs = len(self.testloader.dataset)
        seq_index = np.random.randint(0, num_seqs, (1,))[0]
        seq, _, _, _ = self.testloader.dataset[seq_index]
        seq = torch.from_numpy(seq).float().squeeze()

        print(seq.shape)
        self.visualize_sequence(seq, 'original')

        # setting anchors
        T, H, W = seq.size()
        in_t, in_h, in_w = T, 32, 40
        out_t, out_h, out_w = T, 35, 44
        anchor_t = in_t / 2.0
        anchor_x = 20
        anchor_y = 24

        """ gaussian """
        # makeing grid
        grid_t_i = torch.arange(0, out_t).float().view(-1, 1)
        grid_x_i = torch.arange(0, out_w).float().view(-1, 1)
        grid_y_i = torch.arange(0, out_h).float().view(-1, 1)
        mu_t = anchor_t + (grid_t_i - out_t/2.0) * 0.4
        mu_x = anchor_x + (grid_x_i - out_w/2.0) * 10.0 / 11
        mu_y = anchor_y + (grid_y_i - out_h/2.0) * 10.0 / 11
        c = torch.arange(0, T).float().view(1, -1)
        a = torch.arange(0, W).float().view(1, -1)
        b = torch.arange(0, H).float().view(1, -1)
        transform_t = torch.exp(-1 * torch.pow(c - mu_t, 2) / 2)
        transform_x = torch.exp(-1 * torch.pow(a - mu_x, 2) / 2)
        transform_y = torch.exp(-1 * torch.pow(b - mu_y, 2) / 2)

        # normalize, sum over H and W dims
        eps = 1e-6
        eps_tensor_t = eps * torch.ones(T)
        eps_tensor_h = eps * torch.ones(H)
        eps_tensor_w = eps * torch.ones(W)
        Ft = transform_t / torch.max(torch.sum(transform_t, 1, keepdim=True), eps_tensor_t)
        Fx = transform_x / torch.max(torch.sum(transform_x, 1, keepdim=True), eps_tensor_w)
        Fy = transform_y / torch.max(torch.sum(transform_y, 1, keepdim=True), eps_tensor_h)

        # sampling
        Ftv = Ft
        Fyv = Fy.view(1, Fy.size(0), Fy.size(1))
        Fxv = Fx.view(1, Fx.size(0), Fx.size(1))
        Fxt = torch.transpose(Fxv, 1, 2)
        gaussian_sampled = torch.matmul(Fyv, torch.matmul(seq, Fxt))
        gaussian_sampled = gaussian_sampled.view(gaussian_sampled.size(0), -1)
        gaussian_sampled = torch.matmul(Ftv, gaussian_sampled)
        gaussian_sampled = gaussian_sampled.view(out_t, out_h, out_w)

        # pad top and bottom along the H dim
        x_h = gaussian_sampled.size(1)
        pad_t = (H - x_h) // 2
        pad_b = H - pad_t - x_h
        gaussian_sampled = F.pad(gaussian_sampled, pad=(0, 0, pad_t, pad_b))
        index = mu_t.squeeze().cpu().numpy()
        self.visualize_sequence(gaussian_sampled, 'gaussian', index)

        """ gaussian inverse """
        inv_Ft = transform_t / torch.max(torch.sum(transform_t, 0, keepdim=True), eps_tensor_t)
        inv_Fx = transform_x / torch.max(torch.sum(transform_x, 0, keepdim=True), eps_tensor_w)
        inv_Fy = transform_y / torch.max(torch.sum(transform_y, 0, keepdim=True), eps_tensor_h)
        inv_Ftv = inv_Ft
        inv_Fyv = inv_Fy.view(1, inv_Fy.size(0), inv_Fy.size(1))
        inv_Fxv = inv_Fx.view(1, inv_Fx.size(0), inv_Fx.size(1))
        inv_Ftt = torch.transpose(inv_Ftv, 0, 1)
        inv_Fxt = torch.transpose(inv_Fxv, 1, 2)
        inv_Fyt = torch.transpose(inv_Fyv, 1, 2)
        gaussian_sampled = torch.matmul(Fyv, torch.matmul(seq, Fxt))
        gaussian_sampled = gaussian_sampled.view(gaussian_sampled.size(0), -1)
        gaussian_sampled = torch.matmul(Ftv, gaussian_sampled)
        gaussian_sampled = gaussian_sampled.view(out_t, out_h, out_w)
        gaussian_inv_sampled = torch.matmul(inv_Fyt, torch.matmul(gaussian_sampled, inv_Fxv))
        gaussian_inv_sampled = gaussian_inv_sampled.view(gaussian_inv_sampled.size(0), -1)
        gaussian_inv_sampled = torch.matmul(inv_Ftt, gaussian_inv_sampled)
        gaussian_inv_sampled = gaussian_inv_sampled.view(T, H, W)
        self.visualize_sequence(gaussian_inv_sampled, 'gaussian_inv')


        """ trilinear """
        # making grid
        # normalize the grid in [-1, 1]
        normalized_t = mu_t.squeeze() / (T-1) * 2 - 1
        normalized_x = mu_x.squeeze() / (W-1) * 2 - 1
        normalized_y = mu_y.squeeze() / (H-1) * 2 - 1
        normalized_t = normalized_t[None, :, None, None, None].repeat(1, 1, out_h, out_w, 1)
        normalized_x = normalized_x[None, None, None, :, None].repeat(1, out_t, out_h, 1, 1)
        normalized_y = normalized_y[None, None, :, None, None].repeat(1, out_t, 1, out_w, 1)
        # trilinear_grid = torch.cat([normalized_t, normalized_y, normalized_x], -1)
        trilinear_grid = torch.cat([normalized_x, normalized_y, normalized_t], -1)

        # sampling
        reshaped_seq = seq.view(1, 1, T, H, W)
        trilinear_sampled = F.grid_sample(reshaped_seq, trilinear_grid).squeeze()
        # pad top and bottom along the H dim
        x_h = trilinear_sampled.size(1)
        pad_t = (H - x_h) // 2
        pad_b = H - pad_t - x_h
        trilinear_sampled = F.pad(trilinear_sampled, pad=(0, 0, pad_t, pad_b))
        self.visualize_sequence(trilinear_sampled, 'trilinear', index)

        """ mix """
        # making grid
        normalized_T = mu_t.squeeze() / (T-1) * 2 - 1
        normalized_T = normalized_T[None, :, None, None]
        normalized_S = torch.zeros(1, out_t, 1, 1) - 2
        # mix_grid = torch.cat([normalized_T, normalized_S], -1)
        mix_grid = torch.cat([normalized_S, normalized_T], -1)

        # sampling
        mix_sampled = torch.matmul(Fyv, torch.matmul(seq, Fxt))
        mix_sampled = mix_sampled.view(T, out_h*out_w).permute(1, 0).contiguous()
        mix_sampled = mix_sampled.view(1, out_h*out_w, T, 1)
        mix_sampled = F.grid_sample(mix_sampled, mix_grid)
        mix_sampled = mix_sampled.view(out_h, out_w, out_t)
        mix_sampled = mix_sampled.permute(2, 0, 1).contiguous()
        # pad top and bottom along the H dim
        x_h = mix_sampled.size(1)
        pad_t = (H - x_h) // 2
        pad_b = H - pad_t - x_h
        mix_sampled = F.pad(mix_sampled, pad=(0, 0, pad_t, pad_b))
        self.visualize_sequence(mix_sampled, 'mix', index)

        """ mix inverse """
        mix_inv_sampled = torch.matmul(Fyv, torch.matmul(seq, Fxt))
        mix_inv_sampled = torch.matmul(inv_Fyt, torch.matmul(mix_inv_sampled, inv_Fxv))
        mix_inv_sampled = mix_inv_sampled.view(T, H*W).permute(1, 0).contiguous()
        mix_inv_sampled = mix_inv_sampled.view(1, H*W, T, 1)
        mix_inv_sampled = F.grid_sample(mix_inv_sampled, mix_grid)
        mix_inv_sampled = mix_inv_sampled.view(H, W, out_t)
        mix_inv_sampled = mix_inv_sampled.permute(2, 0, 1).contiguous()
        # pad top and bottom along the H dim
        x_h = mix_inv_sampled.size(1)
        pad_t = (H - x_h) // 2
        pad_b = H - pad_t - x_h
        mix_inv_sampled = F.pad(mix_inv_sampled, pad=(0, 0, pad_t, pad_b))
        self.visualize_sequence(mix_inv_sampled, 'mix_inv', index)


    def debug(self):
        # image = torch.arange(8).view(1, 1, 2, 2, 2).float()
        # grid = torch.Tensor([[1, 0, 1], [-1, 0, 1]]).view(1, 1, 1, 2, 3).float()
        image = torch.arange(2).view(1, 1, 2, 1).float()
        grid = torch.Tensor([[-1, 1], [1, -1]]).view(1, 2, 1, 2).float()
        sampled = F.grid_sample(image, grid)
        print('image:\n', image[0, 0])
        print('grid:\n', grid.squeeze())
        print('sampled:\n', sampled[0, 0])
