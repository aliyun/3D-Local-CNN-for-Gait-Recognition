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
import solvers
from utils import AverageMeter
from solvers import Solver

__all__ = ['GaitSetSolver']


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


# TODO
# 把seq外面包着的list去掉
# 去掉batch_frame
# 优化test速度
class GaitSetSolver(Solver):
    def train(self):
        self.build_data()
        self.build_model()
        self.build_optimizer()
        self.build_loss()
        self.restore_iter = 0

        # Print out configurations
        self.print_log('{} samples in train set'.format(
            len(self.trainloader.dataset)))
        self.print_log('{} samples in test set'.format(
            len(self.testloader.dataset)))
        if self.cfg.print_model:
            self.print_log('Architecture:\n{}'.format(self.model))
            num_params = sum(p.numel() for p in self.model.parameters()
                             if p.requires_grad)
            self.print_log('Parameters: {}'.format(num_params))
        self.print_log('Configurations:\n{}\n'.format(
            json.dumps(vars(self.cfg), indent=4)))

        # load from checkpoints
        if self.cfg.weights is not None:
            self.load_checkpoint(self.cfg.weights)

        self.model.train()
        lossMeter = AverageMeter()
        timeMeter1 = AverageMeter()
        timeMeter2 = AverageMeter()
        end = time.time()

        for seq, view, seq_type, label in self.trainloader:
            timeMeter1.update(time.time() - end)
            end = time.time()

            self.restore_iter += 1
            lr = self.lr_scheduler.step(self.restore_iter)

            for i in range(len(seq)):
                seq[i] = np2var(seq[i]).float()
            # if batch_frame is not None:
            # batch_frame = np2var(batch_frame).int()
            # target_label = [train_label_set.index(l) for l in label]
            target_label = np2var(np.array(label)).long()

            feature = self.model(*seq, None)
            loss = self.loss(feature, target_label)
            lossMeter.update(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            timeMeter2.update(time.time() - end)

            if self.restore_iter % self.cfg.log_interval == 0:
                self.print_log(
                    'Iter: {}'.format(self.restore_iter) +
                    ' - DataTime: {:.0f}s'.format(timeMeter1.sum) +
                    ' - ForwardTime: {:.0f}s'.format(timeMeter2.sum) +
                    ' - Lr: {:.2e}'.format(lr) +
                    ' - Loss: {:.6f}'.format(lossMeter.avg))
                lossMeter.reset()
                timeMeter1.reset()
                timeMeter2.reset()

            if self.restore_iter % self.cfg.save_interval == 0:
                self.save_checkpoint(
                    self.restore_iter, {
                        'iteration': self.restore_iter,
                        'model': self.model.module.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    })

            if self.restore_iter % self.cfg.test_interval == 0:
                self._test()

            if self.restore_iter == self.cfg.num_iter:
                break
            end = time.time()

    def load_checkpoint(self, filename, optim=True):
        state = torch.load(filename)
        self.restore_iter = state['iteration']
        self.model.module.load_state_dict(state['model'])
        if optim:
            self.optimizer.load_state_dict(state['optimizer'])
            self.print_log('Load weights and optim from {}'.format(filename))
        else:
            self.print_log('Load weights from {}'.format(filename))
        return self.restore_iter

    def test(self):
        if self.cfg.weights is None:
            raise ValueError('Please appoint --weights.')
        self.load_checkpoint(self.cfg.weights, optim=False)
        self._test()

    def _test(self):
        self.model.eval()

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        for i, x in enumerate(self.testloader):
            seq, view, seq_type, label = x
            for j in range(len(seq)):
                seq[j] = np2var(seq[j]).float()
            # if batch_frame is not None:
            # batch_frame = np2var(batch_frame).int()

            feature = self.model(*seq, None)
            n, num_bin, _ = feature.size()
            feature_list.append(feature.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list += label

        feature = np.concatenate(feature_list, 0)
        view = view_list
        seq_type = seq_type_list
        label = np.array(label_list)

        view_list = list(set(view))
        view_list.sort()
        view_num = len(view_list)
        sample_num = len(feature)

        probe_seq_dict = {
            'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'],
                      ['cl-01', 'cl-02']],
            'OUMVLP': [['00']]
        }
        gallery_seq_dict = {
            'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
            'OUMVLP': [['01']]
        }

        num_rank = 5
        dataset = 'CASIA' if 'CASIA' in self.cfg.dataset else 'OUMVLP'
        acc = np.zeros(
            [len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
        for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
            for gallery_seq in gallery_seq_dict[dataset]:
                for (v1, probe_view) in enumerate(view_list):
                    for (v2, gallery_view) in enumerate(view_list):
                        gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                            view, [gallery_view])
                        gallery_x = feature[gseq_mask, :]
                        gallery_y = label[gseq_mask]

                        pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                            view, [probe_view])
                        probe_x = feature[pseq_mask, :]
                        probe_y = label[pseq_mask]

                        dist = cuda_dist(probe_x, gallery_x)
                        idx = dist.sort(1)[1].cpu().numpy()

                        out = np.reshape(probe_y, [-1, 1])
                        out = np.cumsum(out == gallery_y[idx[:, 0:num_rank]],
                                        1)
                        out = np.sum(out > 0, 0)
                        out = np.round(out * 100 / dist.shape[0], 2)
                        acc[p, v1, v2, :] = out
                        # acc[p, v1, v2, :] = np.round(np.sum(np.cumsum(np.reshape(
                        # probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]],
                        # 1) > 0, 0) * 100 / dist.shape[0], 2)

        if dataset == 'CASIA':
            # Print rank-1 accuracy of the best model
            # e.g.
            # ===Rank-1 (Include identical-view cases)===
            # NM: 95.405,     BG: 88.284,     CL: 72.041
            self.print_log('===Rank-1 (Include identical-view cases)===')
            self.print_log('NM: %.3f,\tBG: %.3f,\tCL: %.3f' %
                           (np.mean(acc[0, :, :, 0]), np.mean(
                               acc[1, :, :, 0]), np.mean(acc[2, :, :, 0])))

            # self.print_log rank-1 accuracy of the best model，excluding identical-view cases
            # e.g.
            # ===Rank-1 (Exclude identical-view cases)===
            # NM: 94.964,     BG: 87.239,     CL: 70.355
            self.print_log('===Rank-1 (Exclude identical-view cases)===')
            self.print_log('NM: %.3f,\tBG: %.3f,\tCL: %.3f' %
                           (de_diag(acc[0, :, :, 0]), de_diag(
                               acc[1, :, :, 0]), de_diag(acc[2, :, :, 0])))

            # self.print_log rank-1 accuracy of the best model (Each Angle)
            # e.g.
            # ===Rank-1 of each angle (Exclude identical-view cases)===
            # NM: [90.80 97.90 99.40 96.90 93.60 91.70 95.00 97.80 98.90 96.80 85.80]
            # BG: [83.80 91.20 91.80 88.79 83.30 81.00 84.10 90.00 92.20 94.45 79.00]
            # CL: [61.40 75.40 80.70 77.30 72.10 70.10 71.50 73.50 73.50 68.40 50.00]
            # np.set_self.print_logoptions(precision=2, floatmode='fixed')
            np.printoptions(precision=2, floatmode='fixed')
            self.print_log(
                '===Rank-1 of each angle (Exclude identical-view cases)===')
            s = '[' + ', '.join(['{:.3f}' for _ in range(view_num)]) + ']'
            self.print_log('NM: ' + s.format(*de_diag(acc[0, :, :, 0], True)))
            self.print_log('BG: ' + s.format(*de_diag(acc[1, :, :, 0], True)))
            self.print_log('CL: ' + s.format(*de_diag(acc[2, :, :, 0], True)))

        elif dataset == 'OUMVLP':
            self.print_log('===Rank-1 (Include identical-view cases)===')
            self.print_log('{:.3f}'.format(np.mean(acc[0, :, :, 0])))

            self.print_log('===Rank-1 (Exclude identical-view cases)===')
            self.print_log('{:.3f}'.format(de_diag(acc[0, :, :, 0])))

            np.printoptions(precision=2, floatmode='fixed')
            self.print_log(
                '===Rank-1 of each angle (Exclude identical-view cases)===')
            s = '[' + ', '.join(['{:.3f}' for _ in range(view_num)]) + ']'
            self.print_log(s.format(*de_diag(acc[0, :, :, 0], True)))
