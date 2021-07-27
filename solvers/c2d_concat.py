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
from solvers import BaselineSolver, C2D_Local_Solver


class C2D_Local_Concat_Solver(C2D_Local_Solver):
    r""" Independetly calculate loss for local features
    Configs:
        w_local: weights for local feature loss
    """

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
            acc, local_acc = self._test()
            if len(acc) > 1:
                self.writer.add_scalar('test/accNM', acc[0], self.iter)
                self.writer.add_scalar('test/accBG', acc[1], self.iter)
                self.writer.add_scalar('test/accCL', acc[2], self.iter)
                self.writer.add_scalar('test/local_accNM', local_acc[0], self.iter)
                self.writer.add_scalar('test/local_accBG', local_acc[1], self.iter)
                self.writer.add_scalar('test/local_accCL', local_acc[2], self.iter)
            else:
                self.writer.add_scalar('test/acc', acc, self.iter)
                self.writer.add_scalar('test/local_acc', local_acc, self.iter)

        # Meters
        self.best_acc, self.best_iter = [0], -1
        lossMeters = [AverageMeter() for _ in range(4)]
        deltaMeters = [AverageMeter() for _ in range(3)]
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

            output, local_out, deltas = self.model(seq)
            triplet_loss, loss_num, ratio = self.loss(output, label)
            local_loss, local_loss_num, local_ratio = self.loss(local_out, label)

            # delta regularization
            num = len(deltas[0])
            head_reg = torch.norm(deltas[0] - 4.0/11).div(num)
            torso_reg = torch.norm(deltas[1] - 10.0/11).div(num)
            legs_reg = torch.norm(deltas[2] - 10.0/11).div(num)
            reg_loss = self.cfg.w0*head_reg + self.cfg.w1*torso_reg + self.cfg.w2*legs_reg
            loss = triplet_loss + reg_loss + self.cfg.w_local * local_loss

            lossMeters[0].update(triplet_loss.item())
            lossMeters[1].update(loss_num.item())
            lossMeters[2].update(ratio.item())
            lossMeters[3].update(local_loss.item())
            deltaMeters[0].update(deltas[0].mean().item())
            deltaMeters[1].update(deltas[1].mean().item())
            deltaMeters[2].update(deltas[2].mean().item())

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
                               ' - Local: {:.2f}'.format(lossMeters[3].avg) +
                               ' - Num: {:.2e}'.format(lossMeters[1].avg) +
                               ' - Pos/Neg: {:.2f}'.format(lossMeters[2].avg) +
                               ' - Head: {:.2f}'.format(deltaMeters[0].avg) +
                               ' - Torso: {:.2f}'.format(deltaMeters[1].avg) +
                               ' - Legs: {:.2f}'.format(deltaMeters[2].avg))
                self.writer.add_scalar('train/loss', lossMeters[0].avg, self.iter)
                self.writer.add_scalar('train/loss_num', lossMeters[1].avg, self.iter)
                self.writer.add_scalar('train/ratio', lossMeters[2].avg, self.iter)
                self.writer.add_scalar('train/local_loss', lossMeters[3].avg, self.iter)
                self.writer.add_scalar('train/head', deltaMeters[0].avg, self.iter)
                self.writer.add_scalar('train/torso', deltaMeters[1].avg, self.iter)
                self.writer.add_scalar('train/legs', deltaMeters[2].avg, self.iter)

                for m in lossMeters + deltaMeters + [timeMeter1, timeMeter2]:
                    m.reset()

                # show distributions of weights and grads
                self.show_info()

            # save checkpoints
            self.save()

            # test
            if self.iter % self.cfg.test_interval == 0:
                acc, local_acc = self._test()
                self.collect(acc)
                if len(acc) > 1:
                    self.writer.add_scalar('test/accNM', acc[0], self.iter)
                    self.writer.add_scalar('test/accBG', acc[1], self.iter)
                    self.writer.add_scalar('test/accCL', acc[2], self.iter)
                    self.writer.add_scalar('test/local_accNM', local_acc[0], self.iter)
                    self.writer.add_scalar('test/local_accBG', local_acc[1], self.iter)
                    self.writer.add_scalar('test/local_accCL', local_acc[2], self.iter)
                else:
                    self.writer.add_scalar('test/acc', acc, self.iter)
                    self.writer.add_scalar('test/local_acc', local_acc, self.iter)

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

        output_list = list()
        local_out_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        params_dict = {}

        # visualiza fusion module's weight
        # self.vis_fusion_weight()

        # for self.vis_attention
        num_images = len(self.testloader)
        spec_index = getattr(self.cfg, 'spec_index', -1)
        if spec_index == -1:
            spec_index = np.random.randint(0, num_images, (1,))[0]

        for i, x in enumerate(self.testloader):
            seq, view, seq_type, label = x
            seq = seq.float().cuda()

            output, local_out, params, features = self.model(seq)
            n, num_bin, _ = output.size()
            output_list.append(output.view(n, -1).data.cpu().numpy())
            local_out_list.append(local_out.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list.append(label.item())
            params_dict[i] = params

            # visualiza the attention maps of different branches
            if i == spec_index:
                self.vis_attention(features)

        # visualize the distributions of global features and local features
        self.vis_feature_dist(output_list, local_out_list)

        # collect localization parameters and visualize the distribution
        self.vis_loc_param_dist(params_dict)

        acc = self._compute_accuracy(output_list, view_list, seq_type_list,
                                     label_list)
        local_acc = self._compute_accuracy(local_out_list, view_list,
                                           seq_type_list, label_list)
        return acc, local_acc


    def vis_feature_dist(self, gl, local):
        gl = np.concatenate(gl, axis=0)
        local = np.concatenate(local, axis=0)

        self.writer.add_histogram('feature_dist/global', gl, self.iter)
        self.writer.add_histogram('feature_dist/local', local, self.iter)
        return
