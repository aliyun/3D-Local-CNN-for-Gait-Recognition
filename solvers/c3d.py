#! /usr/bin/env python
import os
import pdb
import time
import yaml
import json
import pickle
import random
import shutil
import argparse
import numpy as np
from collections import defaultdict

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# plot
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from utils import AverageMeter, LearningRate, accuracy, LossWeightDecay
from solvers import BaselineSolver

class C3D_Solver(BaselineSolver):

    def build_optimizer(self):
        if self.cfg.optimizer == 'SGD':
            self.optimizer_backbone = self._build_sgd(
                self.model.module.backbone,
                self.model.module.spatial_pool,
                self.model.module.temporal_pool,
                self.model.module.hpm,
            )
            self.optimizer_top = self._build_sgd(
                self.model.module.compact_block,
                self.model.module.classifier
            )

        elif self.cfg.optimizer == 'Adam':
            self.optimizer_backbone = self._build_adam(
                self.model.module.backbone,
                self.model.module.spatial_pool,
                self.model.module.temporal_pool,
                self.model.module.hpm,
            )
            self.optimizer_top = self._build_adam(
                self.model.module.compact_block,
                self.model.module.classifier,
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


    def build_loss(self):
        self.criterion_early = self._build_one_loss(self.cfg.early_loss,
                                                    self.cfg.early_loss_args)
        self.criterion_mid = self._build_one_loss(self.cfg.mid_loss,
                                                  self.cfg.mid_loss_args)
        self.criterion_late = self._build_one_loss(self.cfg.late_loss,
                                                   self.cfg.late_loss_args)
        self.early_loss_weight = LossWeightDecay(**self.cfg.early_loss_weight)
        self.mid_loss_weight = LossWeightDecay(**self.cfg.mid_loss_weight)
        self.late_loss_weight = LossWeightDecay(**self.cfg.late_loss_weight)


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

        # Meters
        self.best_acc, self.best_iter = [0], -1
        meters = defaultdict(lambda: AverageMeter())

        end = time.time()
        for seq, view, seq_type, label in self.trainloader:
            self.model.train()
            meters['dataTime'].update(time.time() - end)
            end = time.time()

            # Learning rate and loss weights decay
            lr_backbone = self.lr_scheduler_backbone.step(self.iter)
            lr_top = self.lr_scheduler_top.step(self.iter)
            lw_early = self.early_loss_weight.step(self.iter)
            lw_mid = self.mid_loss_weight.step(self.iter)
            lw_late = self.late_loss_weight.step(self.iter)
            self.iter += 1

            seq, label = seq.float().cuda(), label.long().cuda()

            # forward and calculate loss
            out1, out2, preds = self.model(seq)
            early_loss, loss_num = self.criterion_early(out1, label)
            mid_loss, mid_acc = self.criterion_mid(out2, label)
            late_loss = self.criterion_late(preds, label)
            prec, = accuracy(preds, label, topk=(1,))
            loss = lw_early*early_loss + lw_mid*mid_loss + lw_late*late_loss

            # backward
            self.optimizer_top.zero_grad()
            self.optimizer_backbone.zero_grad()
            loss.backward()
            self.optimizer_top.step()
            self.optimizer_backbone.step()

            # record loss
            meters['modelTime'].update(time.time() - end)
            meters['earlyLoss'].update(early_loss)
            meters['midLoss'].update(mid_loss)
            meters['lateLoss'].update(late_loss)
            meters['lossNum'].update(loss_num)
            meters['Acc'].update(prec)
            meters['midAcc'].update(mid_acc)

            # show log info
            if self.iter % self.cfg.log_interval == 0:
                self.print_log('Iter: {}/{}'.format(self.iter, self.cfg.num_iter) +
                               ' - Data: {:.0f}s'.format(meters['dataTime'].sum) +
                               ' - Model: {:.0f}s'.format(meters['modelTime'].sum) +
                               ' - Backbone: {:.2e}'.format(lr_backbone) +
                               ' - Top: {:.2e}'.format(lr_top) +
                               ' - W_Early: {:.2f}'.format(lw_early) +
                               ' - W_Mid: {:.2f}'.format(lw_mid) +
                               ' - W_Late: {:.2f}'.format(lw_late) +
                               ' - Num: {:.2e}'.format(meters['lossNum'].avg) +
                               ' - Loss_Mid: {:.2f}'.format(meters['midLoss'].avg) +
                               ' - Loss_Late: {:.2f}'.format(meters['lateLoss'].avg) +
                               ' - MidAcc: {:.2%}'.format(meters['midAcc'].avg) +
                               ' - Acc: {:.2%}'.format(meters['Acc'].avg))

                for i in ['earlyLoss', 'lossNum', 'midLoss', 'lateLoss',
                          'midAcc', 'Acc']:
                    self.writer.add_scalar('train/{}'.format(i), meters[i].avg, self.iter)

                for m in meters.values():
                    m.reset()

                # show distributions of weights and grads
                self.show_info()

            # save checkpoints
            self.save()

            # test
            if self.iter % self.cfg.test_interval == 0:
                acc = self._test()
                self.collect(acc)

            if self.iter == self.cfg.num_iter:
                self.print_log('\nBest Acc: {}'.format(self.best_acc) +
                               '\nIter: {}'.format(self.best_iter) +
                               '\nDir: {}'.format(self.work_dir) +
                               '\nTime: {}'.format(
                                   self._convert_time(time.time() - start_time)))
                return
            end = time.time()

    def _test(self):
        self.model.eval()

        feature_list1 = list()
        feature_list2 = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        for i, x in enumerate(self.testloader):
            seq, view, seq_type, label = x
            seq = seq.float().cuda()

            out1, out2 = self.model(seq)
            n = out1.size(0)
            feature_list1.append(out1.view(n, -1).data.cpu().numpy())
            feature_list2.append(out2.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list.append(label.item())

        self.print_log('Full Euclidean')
        acc_full_euc = self._compute_accuracy(feature_list1, view_list, seq_type_list,
                                              label_list, metric='euclidean')
        self.print_log('Compact Euclidean')
        acc_compact_euc = self._compute_accuracy(feature_list2, view_list, seq_type_list,
                                                 label_list, metric='euclidean')
        self.print_log('Full Cosine')
        acc_full_cos = self._compute_accuracy(feature_list1, view_list, seq_type_list,
                                              label_list, metric='cosine')
        self.print_log('Compact Cosine')
        acc_compact_cos = self._compute_accuracy(feature_list2, view_list, seq_type_list,
                                                 label_list, metric='cosine')

        if len(acc_compact_euc) > 1:
            self.writer.add_scalar('test_fullEuc/AccNM', acc_full_euc[0], self.iter)
            self.writer.add_scalar('test_fullEuc/AccBG', acc_full_euc[1], self.iter)
            self.writer.add_scalar('test_fullEuc/AccCL', acc_full_euc[2], self.iter)
            self.writer.add_scalar('test_compactEuc/AccNM', acc_compact_euc[0], self.iter)
            self.writer.add_scalar('test_compactEuc/AccBG', acc_compact_euc[1], self.iter)
            self.writer.add_scalar('test_compactEuc/AccCL', acc_compact_euc[2], self.iter)
            self.writer.add_scalar('test_fullCos/AccNM', acc_full_cos[0], self.iter)
            self.writer.add_scalar('test_fullCos/AccBG', acc_full_cos[1], self.iter)
            self.writer.add_scalar('test_fullCos/AccCL', acc_full_cos[2], self.iter)
            self.writer.add_scalar('test_compactCos/AccNM', acc_compact_cos[0], self.iter)
            self.writer.add_scalar('test_compactCos/AccBG', acc_compact_cos[1], self.iter)
            self.writer.add_scalar('test_compactCos/AccCL', acc_compact_cos[2], self.iter)
        else:
            self.writer.add_scalar('test/fullEucAcc', acc_full_euc[0], self.iter)
            self.writer.add_scalar('test/compactEucAcc', acc_compact_euc[0], self.iter)
            self.writer.add_scalar('test/fullCosAcc', acc_full_cos[0], self.iter)
            self.writer.add_scalar('test/compactCosAcc', acc_compact_cos[0], self.iter)
        target_acc = getattr(self.cfg, 'target_acc', 'full_euc')
        accs = {'full_euc': acc_full_euc, 'full_cos': acc_full_cos,
                'compact_euc': acc_compact_euc, 'compact_cos': acc_compact_cos}
        return accs[target_acc]

    def all_test(self):
        self.build_data()
        self.build_model()

        if self.cfg.pretrained is None:
            raise ValueError('Please appoint --pretrained.')
        self.load_checkpoint(self.cfg.pretrained, optim=False)
        self.model.eval()

        feature_list1 = list()
        feature_list2 = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        for i, x in enumerate(self.testloader):
            seq, view, seq_type, label = x
            seq = seq.float().cuda()

            out1, out2 = self.model(seq)
            n = out1.size(0)
            feature_list1.append(out1.view(n, -1).data.cpu().numpy())
            feature_list2.append(out2.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list.append(label.item())

        self.print_log('Full Euclidean')
        acc1 = self._compute_accuracy(feature_list1, view_list, seq_type_list,
                                      label_list, metric='euclidean')
        self.print_log('Compact Euclidean')
        acc2 = self._compute_accuracy(feature_list2, view_list, seq_type_list,
                                      label_list, metric='euclidean')
        self.print_log('Full Cosine')
        acc3 = self._compute_accuracy(feature_list1, view_list, seq_type_list,
                                      label_list, metric='cosine')
        self.print_log('Compact Cosine')
        acc4 = self._compute_accuracy(feature_list2, view_list, seq_type_list,
                                      label_list, metric='cosine')
