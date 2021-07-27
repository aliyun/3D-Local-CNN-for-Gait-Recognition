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


class C2D_Local_3Branch_Fix_Solver(Solver):

    def build_optimizer(self):
        if self.cfg.optimizer == 'SGD':
            self.optimizer_backbone = self._build_sgd(
                self.model.module.backbone.layer1,
                self.model.module.backbone.layer2,
                self.model.module.backbone.layer3,
                self.model.module.backbone.layer4,
                self.model.module.backbone.layer5,
                self.model.module.backbone.layer6
            )
            self.optimizer_top = self._build_sgd(
                self.model.module.backbone.head,
                self.model.module.backbone.torso,
                self.model.module.backbone.legs,
                self.model.module.backbone.feature_fusion,
                self.model.module.spatial_pool,
                self.model.module.temporal_pool,
                self.model.module.classifier
            )

        elif self.cfg.optimizer == 'Adam':
            self.optimizer_backbone = self._build_adam(
                self.model.module.backbone.layer1,
                self.model.module.backbone.layer2,
                self.model.module.backbone.layer3,
                self.model.module.backbone.layer4,
                self.model.module.backbone.layer5,
                self.model.module.backbone.layer6
            )
            self.optimizer_top = self._build_adam(
                self.model.module.backbone.head,
                self.model.module.backbone.torso,
                self.model.module.backbone.legs,
                self.model.module.backbone.feature_fusion,
                self.model.module.spatial_pool,
                self.model.module.temporal_pool,
                self.model.module.classifier
            )

        else:
            raise ValueError()
        self.lr_scheduler_backbone = LearningRate(self.optimizer_backbone,
                                                  **self.cfg.lr_decay_backbone)
        self.lr_scheduler_top = LearningRate(self.optimizer_top,
                                                  **self.cfg.lr_decay_top)


    def save_checkpoint(self, filename):
        state = {
            'iteration': self.iter,
            'model': self.model.module.state_dict(),
            'optimizer_backbone': self.optimizer_backbone.state_dict(),
            'optimizer_top': self.optimizer_top.state_dict(),
        }
        torch.save(state, filename)
        self.print_log('Save checkpoint to {}'.format(filename))
        return self.iter

    def load_checkpoint(self, filename, optim=True):
        state = torch.load(filename)
        iter = state['iteration']
        self.model.module.load_state_dict(state['model'])
        if optim:
            self.optimizer_backbone.load_state_dict(state['optimizer_backbone'])
            self.optimizer_top.load_state_dict(state['optimizer_top'])
            self.print_log('Load weights and optim from {}'.format(filename))
        else:
            self.print_log('Load weights from {}'.format(filename))
        return iter


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

        # Test before training
        if self.cfg.test_first:
            acc = self._test()
            if len(acc) > 1:
                self.writer.add_scalar('test/accNM', acc[0], self.iter)
                self.writer.add_scalar('test/accBG', acc[1], self.iter)
                self.writer.add_scalar('test/accCL', acc[2], self.iter)
            else:
                self.writer.add_scalar('test/acc', acc, self.iter)

        self.model.train()
        self.best_acc, self.best_iter = [0], -1

        lossMeter = AverageMeter()
        timeMeter1, timeMeter2 = AverageMeter(), AverageMeter()

        end = time.time()
        for seq, view, seq_type, label in self.trainloader:
            timeMeter1.update(time.time() - end)
            end = time.time()

            lr_backbone = self.lr_scheduler_backbone.step(self.iter)
            lr_top = self.lr_scheduler_top.step(self.iter)
            self.iter += 1

            seq, label = seq.float().cuda(), label.long().cuda()

            feature = self.model(seq)
            loss = self.loss(feature, label)
            lossMeter.update(loss.item())

            self.optimizer_backbone.zero_grad()
            self.optimizer_top.zero_grad()
            loss.backward()
            self.optimizer_backbone.step()
            self.optimizer_top.step()

            timeMeter2.update(time.time() - end)

            if self.iter % self.cfg.log_interval == 0:
                self.print_log('Iter: {}'.format(self.iter) +
                               ' - Data: {:.0f}s'.format(timeMeter1.sum) +
                               ' - Model: {:.0f}s'.format(timeMeter2.sum) +
                               ' - Lr_Backbone: {:.2e}'.format(lr_backbone) +
                               ' - Lr_Top: {:.2e}'.format(lr_top) +
                               ' - Loss: {:.6f}'.format(lossMeter.avg))
                self.writer.add_scalar('train/loss', lossMeter.avg, self.iter)
                lossMeter.reset()
                timeMeter1.reset()
                timeMeter2.reset()

            self.save()

            if self.iter % self.cfg.test_interval == 0:
                acc = self._test()
                self.collect(acc)
                if len(acc) > 1:
                    self.writer.add_scalar('test/accNM', acc[0], self.iter)
                    self.writer.add_scalar('test/accBG', acc[1], self.iter)
                    self.writer.add_scalar('test/accCL', acc[2], self.iter)
                else:
                    self.writer.add_scalar('test/acc', acc, self.iter)

            if self.iter == self.cfg.num_iter:
                self.print_log('\nBest Acc: {}'.format(self.best_acc) +
                               '\nIter: {}'.format(self.best_iter) +
                               '\nDir: {}'.format(self.work_dir) +
                               '\nTime: {}'.format(
                                   self._convert_time(time.time() - start_time)))
                return
            end = time.time()


    def collect(self, acc):
        acc_avg = sum(acc) / len(acc)
        best_avg = sum(self.best_acc) / len(self.best_acc)
        if acc_avg > best_avg:
            self.best_acc = acc
            self.best_iter = self.iter


    def test(self):
        self.build_data()
        self.build_model()

        if self.cfg.pretrained is None:
            raise ValueError('Please appoint --pretrained.')
        self.load_checkpoint(self.cfg.pretrained, optim=False)
        return self._test()

    def _test(self):
        self.model.eval()

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        # visualiza fusion module's weight
        self.collect_weight()

        for i, x in enumerate(self.testloader):
            seq, view, seq_type, label = x
            seq = seq.float().cuda()

            feature = self.model(seq)
            n, num_bin, _ = feature.size()
            feature_list.append(feature.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list.append(label.item())

        feature = np.concatenate(feature_list, 0)
        view = view_list
        seq_type = seq_type_list
        label = np.array(label_list)

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

                        dist = cuda_dist(probe_x, gallery_x)
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


    def collect_weight(self):
        weight = self.model.module.backbone.feature_fusion[0].weight.data
        weight = weight.squeeze().detach().cpu().numpy()
        self.writer.add_image('debug/fusion_weight/{}'.format(self.iter),
                              weight, self.iter, dataformats='HW')
        return


    def collect_distribution(self, params):
        self.model.eval()
        distributions = {}
        # head, torso, legs = params

        # head
        # dx, dy, sigma, delta, gamma, Fx, Fy = head
        # distributions['head_dx'] = dx.squeeze().detach().cpu().numpy()
        # distributions['head_dy'] = dy.squeeze().detach().cpu().numpy()
        # distributions['head_sigma'] = sigma.squeeze().detach().cpu().numpy()
        # distributions['head_delta'] = delta.squeeze().detach().cpu().numpy()
        # distributions['head_gamma'] = gamma.squeeze().detach().cpu().numpy()

        # torso
        # dx, dy, sigma, delta, gamma, Fx, Fy = torso
        dx, dy, sigma, delta, gamma, Fx, Fy = params
        distributions['torso_dx'] = dx.squeeze().detach().cpu().numpy()
        distributions['torso_dy'] = dy.squeeze().detach().cpu().numpy()
        distributions['torso_sigma'] = sigma.squeeze().detach().cpu().numpy()
        distributions['torso_delta'] = delta.squeeze().detach().cpu().numpy()
        distributions['torso_gamma'] = gamma.squeeze().detach().cpu().numpy()

        # legs
        # dx, dy, sigma, delta, gamma, Fx, Fy = legs
        # distributions['legs_dx'] = dx.squeeze().detach().cpu().numpy()
        # distributions['legs_dy'] = dy.squeeze().detach().cpu().numpy()
        # distributions['legs_sigma'] = sigma.squeeze().detach().cpu().numpy()
        # distributions['legs_delta'] = delta.squeeze().detach().cpu().numpy()
        # distributions['legs_gamma'] = gamma.squeeze().detach().cpu().numpy()

        for k, v in distributions.items():
            self.writer.add_histogram('debug/{}'.format(k), v, self.iter)
        return


    def debug_localization(self):
        def f(w, name):
            w = w.squeeze().detach().cpu().numpy()
            sns.distplot(w, hist=True, kde=False, norm_hist=False)
            directory = os.path.join(self.cfg.work_dir, 'figs')
            os.makedirs(directory, exist_ok=True)
            save_path = os.path.join(directory, '{}.pdf'.format(name))
            plt.savefig(save_path)
            print('Saving to {}'.format(save_path))
            plt.close()

        self.build_data()
        self.build_model()

        if self.cfg.pretrained is None:
            raise ValueError('Please appoint --pretrained.')
        self.load_checkpoint(self.cfg.pretrained, optim=False)
        self.model.eval()
        for x in self.testloader:
            seq, view, seq_type, label = x
            seq = seq.float().cuda()

            params = self.model.module.debug(seq)
            head, torso, legs = params

            # head
            dx, dy, sigma2, delta, gamma, Fx, Fy = head
            f(dx, 'head_dx')
            f(dy, 'head_dy')
            f(sigma2, 'head_sigma')
            f(delta, 'head_delta')
            f(gamma, 'head_gamma')

            # torso
            dx, dy, sigma2, delta, gamma, Fx, Fy = torso
            f(dx, 'torso_dx')
            f(dy, 'torso_dy')
            f(sigma2, 'torso_sigma')
            f(delta, 'torso_delta')
            f(gamma, 'torso_gamma')

            # legs
            dx, dy, sigma2, delta, gamma, Fx, Fy = legs
            f(dx, 'legs_dx')
            f(dy, 'legs_dy')
            f(sigma2, 'legs_sigma')
            f(delta, 'legs_delta')
            f(gamma, 'legs_gamma')

            break
