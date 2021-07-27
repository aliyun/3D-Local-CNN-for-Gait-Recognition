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

from utils import AverageMeter, import_class, LearningRate, init_seed
from solvers import BaselineSolver


class C2D_Load_Solver(BaselineSolver):
    r""" Optimize backbone and top differently """
    def build_optimizer(self):
        if self.cfg.optimizer == 'SGD':
            self.optimizer_backbone = self._build_sgd(
                self.model.module.backbone,
            )
            self.optimizer_top = self._build_sgd(
                self.model.module.spatial_pool,
                self.model.module.temporal_pool,
                self.model.module.classifier
            )

        elif self.cfg.optimizer == 'Adam':
            self.optimizer_backbone = self._build_adam(
                self.model.module.backbone,
            )
            self.optimizer_top = self._build_adam(
                self.model.module.spatial_pool,
                self.model.module.temporal_pool,
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
        if self.cfg.test_before_train:
            acc = self._test()
            if len(acc) > 1:
                self.writer.add_scalar('test/accNM', acc[0], self.iter)
                self.writer.add_scalar('test/accBG', acc[1], self.iter)
                self.writer.add_scalar('test/accCL', acc[2], self.iter)
            else:
                self.writer.add_scalar('test/acc', acc, self.iter)

        # Meters
        self.best_acc, self.best_iter = [0], -1
        lossMeters = [AverageMeter() for _ in range(3)]
        timeMeter1, timeMeter2 = AverageMeter(), AverageMeter()

        end = time.time()
        for seq, view, seq_type, label in self.trainloader:
            self.model.train()
            timeMeter1.update(time.time() - end)
            end = time.time()

            lr_backbone = self.lr_scheduler_backbone.step(self.iter)
            lr_top = self.lr_scheduler_top.step(self.iter)
            self.iter += 1

            seq, label = seq.float().cuda(), label.long().cuda()

            feature = self.model(seq)
            loss, loss_num, ratio = self.loss(feature, label)
            lossMeters[0].update(loss.item())
            lossMeters[1].update(loss_num.item())
            lossMeters[2].update(ratio.item())

            self.optimizer_backbone.zero_grad()
            self.optimizer_top.zero_grad()
            loss.backward()
            self.optimizer_backbone.step()
            self.optimizer_top.step()

            timeMeter2.update(time.time() - end)

            # show log info
            if self.iter % self.cfg.log_interval == 0:
                self.print_log('Iter: {}'.format(self.iter) +
                               ' - Data: {:.0f}s'.format(timeMeter1.sum) +
                               ' - Model: {:.0f}s'.format(timeMeter2.sum) +
                               ' - Lr_Backbone: {:.2e}'.format(lr_backbone) +
                               ' - Lr_Top: {:.2e}'.format(lr_top) +
                               ' - Loss: {:.2f}'.format(lossMeters[0].avg) +
                               ' - Num: {:.2e}'.format(lossMeters[1].avg) +
                               ' - Pos/Neg: {:.2f}'.format(lossMeters[2].avg))
                self.writer.add_scalar('train/loss', lossMeters[0].avg, self.iter)
                self.writer.add_scalar('train/loss_num', lossMeters[1].avg, self.iter)
                self.writer.add_scalar('train/ratio', lossMeters[2].avg, self.iter)

                for m in lossMeters + [timeMeter1, timeMeter2]:
                    m.reset()

            # save checkpoints
            self.save()

            # test
            if self.iter % self.cfg.test_interval == 0:
                acc = self._test()
                self.collect(acc)
                if len(acc) > 1:
                    self.writer.add_scalar('test/accNM', acc[0], self.iter)
                    self.writer.add_scalar('test/accBG', acc[1], self.iter)
                    self.writer.add_scalar('test/accCL', acc[2], self.iter)
                else:
                    self.writer.add_scalar('test/acc', acc, self.iter)

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


