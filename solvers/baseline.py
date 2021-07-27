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
from collections import defaultdict

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import AverageMeter
from solvers import Solver

__all__ = ['BaselineSolver']


def euclidean_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist

def cosine_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    return 1 - F.cosine_similarity(x[:, None, :], y[None, :, :], 2)


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    view_num = acc.shape[0]
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / (view_num - 1)
    if not each_angle:
        result = np.mean(result)
    return result


class BaselineSolver(Solver):

    def train(self):
        self.build_data()
        self.build_model()
        self.build_optimizer()
        self.build_loss()
        start_time = time.time()
        self.iter = 0

        # Print out configurations
        self.print_log('{} samples in train set'.format(
            len(self.trainloader.dataset)))
        self.print_log('{} samples in test set'.format(
            len(self.testloader.dataset)))
        if self.cfg.print_model:
            self.print_log('Architecture:\n{}'.format(self.model))
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log('Parameters: {}'.format(num_params))
        self.print_log('Configurations:\n{}\n'.format(
            json.dumps(vars(self.cfg), indent=4)))

        # Load from previous checkpoints
        self.load()

        self.best_acc, self.best_iter = [0], -1
        meters = defaultdict(lambda: AverageMeter())

        end = time.time()
        for seq, view, seq_type, label in self.trainloader:
            self.model.train()
            meters['dataTime'].update(time.time() - end)
            end = time.time()

            lr = self.lr_scheduler.step(self.iter)
            self.iter += 1

            seq, label = seq.float().cuda(), label.long().cuda()

            feature = self.model(seq)
            loss, loss_num = self.loss(feature, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record loss
            meters['modelTime'].update(time.time() - end)
            meters['loss'].update(loss)
            meters['lossNum'].update(loss_num)

            # show log info
            if self.iter % self.cfg.log_interval == 0:
                self.print_log('Iter: {}/{}'.format(self.iter, self.cfg.num_iter) +
                               ' - Data: {:.0f}s'.format(meters['dataTime'].sum) +
                               ' - Model: {:.0f}s'.format(meters['modelTime'].sum) +
                               ' - Lr: {:.2e}'.format(lr) +
                               ' - Loss: {:.2f}'.format(meters['loss'].avg) +
                               ' - Num: {:.2e}'.format(meters['lossNum'].avg))
                for i in ['loss', 'lossNum']:
                    self.writer.add_scalar('train/{}'.format(i), meters[i].avg, self.iter)

                for m in meters.values():
                    m.reset()

            # save checkpoints
            self.save()

            # test
            if self.iter % self.cfg.test_interval == 0:
                acc = self._test()
                self.collect(acc)

                # show distributions of weights and grads
                self.show_info()

            # End
            if self.iter == self.cfg.num_iter:
                self.print_log('\nBest Acc: {}'.format(self.best_acc) +
                               '\nIter: {}'.format(self.best_iter) +
                               '\nDir: {}'.format(self.work_dir) +
                               '\nTime: {}'.format(
                                   self._convert_time(time.time() - start_time)))
                return
            end = time.time()


    def show_info(self, with_weight=True, with_grad=True):
        if with_weight:
            for name, param in self.model.named_parameters():
                w = param.data.cpu().numpy()
                self.writer.add_histogram('weight_info/{}'.format(name), w, self.iter)
        if with_grad:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    w = param.grad.cpu().numpy()
                    self.writer.add_histogram('grad_info/{}'.format(name), w, self.iter)


    def collect(self, acc):
        acc_avg = sum(acc) / len(acc)
        best_avg = sum(self.best_acc) / len(self.best_acc)
        if acc_avg > best_avg:
            self.best_acc = acc
            self.best_iter = self.iter
            # save the best
            path = os.path.join(self.work_dir, self.cfg.save_name+'.pth.tar')
            return self.save_checkpoint(path)


    def test(self):
        self.build_data()
        self.build_model()

        if self.cfg.pretrained is None:
            raise ValueError('Please appoint --pretrained.')
        self.iter = self.load_checkpoint(self.cfg.pretrained, optim=False)
        return self._test()

    def _test(self):
        self.model.eval()

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        for i, x in enumerate(self.testloader):
            seq, view, seq_type, label = x
            seq = seq.float().cuda()

            feature = self.model(seq)
            n = feature.size(0)
            feature_list.append(feature.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list.append(label.item())

        acc = self._compute_accuracy(feature_list, view_list, seq_type_list,
                                     label_list)
        if len(acc) > 1:
            self.writer.add_scalar('test/accNM', acc[0], self.iter)
            self.writer.add_scalar('test/accBG', acc[1], self.iter)
            self.writer.add_scalar('test/accCL', acc[2], self.iter)
        else:
            self.writer.add_scalar('test/acc', acc[0], self.iter)
        return acc


    def _compute_accuracy(self, feature, view, seq_type, label,
                          metric='euclidean'):
        _metrics = {'euclidean': euclidean_dist, 'cosine': cosine_dist}
        dist_metric = _metrics[metric]
        feature = np.concatenate(feature, 0)
        label = np.array(label)

        view_list = list(set(view))
        view_list.sort()
        view_num = len(view_list)
        sample_num = len(feature)

        probe_seq_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                          'OUMVLP': [['00']]}
        gallery_seq_dict = {'CASIA': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                            'OUMVLP': [['01']]}

        num_rank = 5
        dataset = 'CASIA' if 'CASIA' in self.cfg.dataset else 'OUMVLP'
        acc = np.zeros([len(probe_seq_dict[dataset]), view_num, view_num, num_rank])
        for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
            for gallery_seq in gallery_seq_dict[dataset]:
                for (v1, probe_view) in enumerate(view_list):
                    for (v2, gallery_view) in enumerate(view_list):
                        gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(view, [gallery_view])
                        gallery_x = feature[gseq_mask, :]
                        gallery_y = label[gseq_mask]

                        pseq_mask = np.isin(seq_type, probe_seq) & np.isin(view, [probe_view])
                        probe_x = feature[pseq_mask, :]
                        probe_y = label[pseq_mask]

                        dist = dist_metric(probe_x, gallery_x)
                        idx = dist.sort(1)[1].cpu().numpy()

                        out = np.reshape(probe_y, [-1, 1])
                        out = np.cumsum(out == gallery_y[idx[:, 0:num_rank]], 1)
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
            self.print_log('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                np.mean(acc[0, :, :, 0]),
                np.mean(acc[1, :, :, 0]),
                np.mean(acc[2, :, :, 0])))

            # self.print_log rank-1 accuracy of the best model，excluding identical-view cases
            # e.g.
            # ===Rank-1 (Exclude identical-view cases)===
            # NM: 94.964,     BG: 87.239,     CL: 70.355
            self.print_log('===Rank-1 (Exclude identical-view cases)===')
            acc0 = de_diag(acc[0, :, :, 0])
            acc1 = de_diag(acc[1, :, :, 0])
            acc2 = de_diag(acc[2, :, :, 0])
            self.print_log('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                acc0, acc1, acc2))

            # self.print_log rank-1 accuracy of the best model (Each Angle)
            # e.g.
            # ===Rank-1 of each angle (Exclude identical-view cases)===
            # NM: [90.80 97.90 99.40 96.90 93.60 91.70 95.00 97.80 98.90 96.80 85.80]
            # BG: [83.80 91.20 91.80 88.79 83.30 81.00 84.10 90.00 92.20 94.45 79.00]
            # CL: [61.40 75.40 80.70 77.30 72.10 70.10 71.50 73.50 73.50 68.40 50.00]
            # np.set_self.print_logoptions(precision=2, floatmode='fixed')
            np.printoptions(precision=2, floatmode='fixed')
            self.print_log('===Rank-1 of each angle (Exclude identical-view cases)===')
            s = '[' + ', '.join(['{:.3f}' for _ in range(view_num)]) + ']'
            self.print_log('NM: ' + s.format(*de_diag(acc[0, :, :, 0], True)))
            self.print_log('BG: ' + s.format(*de_diag(acc[1, :, :, 0], True)))
            self.print_log('CL: ' + s.format(*de_diag(acc[2, :, :, 0], True)))

            return [acc0, acc1, acc2]

        elif dataset == 'OUMVLP':
            self.print_log('===Rank-1 (Include identical-view cases)===')
            self.print_log('{:.3f}'.format(np.mean(acc[0, :, :, 0])))

            self.print_log('===Rank-1 (Exclude identical-view cases)===')
            self.print_log('{:.3f}'.format(de_diag(acc[0, :, :, 0])))

            np.printoptions(precision=2, floatmode='fixed')
            self.print_log('===Rank-1 of each angle (Exclude identical-view cases)===')
            s = '[' + ', '.join(['{:.3f}' for _ in range(view_num)]) + ']'
            self.print_log(s.format(*de_diag(acc[0, :, :, 0], True)))

            return [de_diag(acc[0, :, :, 0])]

