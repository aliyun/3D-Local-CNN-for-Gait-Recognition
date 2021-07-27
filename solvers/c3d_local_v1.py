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


class Local3dSolverV1(BaselineSolver):

    def build_optimizer(self):
        if self.cfg.optimizer == 'SGD':
            self.optimizer_backbone = self._build_sgd(
                self.model.module.backbone,
            )
            self.optimizer_top = self._build_sgd(
                self.model.module.spatial_pool,
                self.model.module.temporal_pool,
                self.model.module.hpm,
            )
            self.optimizer_local = self._build_sgd(
                self.model.module.local,
            )
            self.optimizer_compact = self._build_sgd(
                self.model.module.compact_block,
                self.model.module.classifier,
            )

        elif self.cfg.optimizer == 'Adam':
            self.optimizer_backbone = self._build_adam(
                self.model.module.backbone,
            )
            self.optimizer_top = self._build_adam(
                self.model.module.spatial_pool,
                self.model.module.temporal_pool,
                self.model.module.hpm,
            )
            self.optimizer_local = self._build_adam(
                self.model.module.local,
            )
            self.optimizer_compact = self._build_adam(
                self.model.module.compact_block,
                self.model.module.classifier,
            )

        else:
            raise ValueError()
        self.lr_scheduler_backbone = LearningRate(self.optimizer_backbone,
                                                  **self.cfg.lr_decay_backbone)
        self.lr_scheduler_top = LearningRate(self.optimizer_top,
                                             **self.cfg.lr_decay_top)
        self.lr_scheduler_local = LearningRate(self.optimizer_local,
                                             **self.cfg.lr_decay_local)
        self.lr_scheduler_compact = LearningRate(self.optimizer_compact,
                                             **self.cfg.lr_decay_compact)

    def save_checkpoint(self, filename):
        state = {
            'iteration': self.iter,
            'model': self.model.module.state_dict(),
            'optimizer_backbone': self.optimizer_backbone.state_dict(),
            'optimizer_top': self.optimizer_top.state_dict(),
            'optimizer_local': self.optimizer_local.state_dict(),
            'optimizer_compact': self.optimizer_compact.state_dict(),
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
            self.optimizer_local.load_state_dict(state['optimizer_local'])
            self.optimizer_compact.load_state_dict(state['optimizer_compact'])
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
        self.local_loss_weight = LossWeightDecay(**self.cfg.local_loss_weight)
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

        # Test before training
        if self.cfg.test_before_train:
            self._test()

        # Meters
        self.best_acc, self.best_iter = [0], -1
        meters = defaultdict(lambda: AverageMeter())

        end = time.time()
        for seq, view, seq_type, label in self.trainloader:
            self.model.train()
            meters['dataTime'].update(time.time() - end)
            end = time.time()

            lr_backbone = self.lr_scheduler_backbone.step(self.iter)
            lr_top = self.lr_scheduler_top.step(self.iter)
            lr_local = self.lr_scheduler_local.step(self.iter)
            lr_compact = self.lr_scheduler_compact.step(self.iter)

            lw_early = self.early_loss_weight.step(self.iter)
            lw_local = self.local_loss_weight.step(self.iter)
            lw_mid = self.mid_loss_weight.step(self.iter)
            lw_late = self.late_loss_weight.step(self.iter)

            self.iter += 1
            seq, label = seq.float().cuda(), label.long().cuda()

            # forward and calculate loss
            feat_global, feat_local, feat_compact, preds, deltas = self.model(seq)
            early_loss, loss_num = self.criterion_early(feat_global, label)
            local_loss, local_loss_num = self.criterion_early(feat_local, label)
            mid_loss, mid_acc = self.criterion_mid(feat_compact, label)
            late_loss = self.criterion_late(preds, label)

            # delta regularization
            num = len(deltas[0])
            head_reg = torch.norm(deltas[0] - 4.0/11).div(num)
            torso_reg = torch.norm(deltas[1] - 10.0/11).div(num)
            legs_reg = torch.norm(deltas[2] - 10.0/11).div(num)
            t_reg = (torch.norm(deltas[3] - 1.0).div(num) + \
                     torch.norm(deltas[4] - 1.0).div(num) + \
                     torch.norm(deltas[5] - 1.0).div(num)) / 3
            reg_loss = self.cfg.w0*head_reg + self.cfg.w1*torso_reg + \
                    self.cfg.w2*legs_reg + self.cfg.w3*t_reg
            loss = lw_early*early_loss + lw_local*local_loss + lw_mid*mid_loss + lw_late*late_loss + reg_loss
            prec, = accuracy(preds, label, topk=(1,))

            # backward
            self.optimizer_backbone.zero_grad()
            self.optimizer_top.zero_grad()
            self.optimizer_local.zero_grad()
            self.optimizer_compact.zero_grad()
            loss.backward()
            self.optimizer_backbone.step()
            self.optimizer_top.step()
            self.optimizer_local.step()
            self.optimizer_compact.step()

            meters['modelTime'].update(time.time() - end)
            meters['earlyLoss'].update(early_loss)
            meters['localLoss'].update(early_loss)
            meters['midLoss'].update(mid_loss)
            meters['lateLoss'].update(late_loss)
            meters['lossNum'].update(loss_num)
            meters['localNum'].update(local_loss_num)
            meters['Acc'].update(prec)
            meters['midAcc'].update(mid_acc)
            meters['deltaHead'].update(deltas[0].mean())
            meters['deltaTorso'].update(deltas[1].mean())
            meters['deltaLegs'].update(deltas[2].mean())
            meters['deltaTimeH'].update(deltas[3].mean())
            meters['deltaTimeT'].update(deltas[4].mean())
            meters['deltaTimeL'].update(deltas[5].mean())

            # show log info
            if self.iter % self.cfg.log_interval == 0:
                self.print_log('Iter: {}'.format(self.iter) +
                               ' - Data: {:.0f}s'.format(meters['dataTime'].sum) +
                               ' - Model: {:.0f}s'.format(meters['modelTime'].sum) +
                               ' - Backbone: {:.2e}'.format(lr_backbone) +
                               ' - Top: {:.2e}'.format(lr_top) +
                               ' - Local: {:.2e}'.format(lr_local) +
                               ' - Compact: {:.2e}'.format(lr_backbone) +
                               ' - Early: {:.2f}'.format(lw_early) +
                               ' - Local: {:.2f}'.format(lw_local) +
                               ' - Mid: {:.2f}'.format(lw_mid) +
                               ' - Late: {:.2f}'.format(lw_late) +
                               ' - Num: {:.2e}'.format(meters['lossNum'].avg) +
                               ' - localNum: {:.2e}'.format(meters['localNum'].avg) +
                               ' - MidAcc: {:.2%}'.format(meters['midAcc'].avg) +
                               ' - Acc: {:.2%}'.format(meters['Acc'].avg) +
                               ' - Head: {:.2f}'.format(meters['deltaHead'].avg) +
                               ' - Torso: {:.2f}'.format(meters['deltaTorso'].avg) +
                               ' - Legs: {:.2f}'.format(meters['deltaLegs'].avg) +
                               ' - TimeH: {:.2f}'.format(meters['deltaTimeH'].avg) +
                               ' - TimeT: {:.2f}'.format(meters['deltaTimeT'].avg) +
                               ' - TimeL: {:.2f}'.format(meters['deltaTimeL'].avg))
                for i in ['earlyLoss', 'localLoss', 'midLoss', 'lateLoss',
                          'lossNum', 'localNum', 'midAcc', 'Acc',
                          'deltaHead', 'deltaTorso', 'deltaLegs',
                          'deltaTimeH', 'deltaTimeT', 'deltaTimeL']:
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

        full_feat_list = list()
        local_feat_list = list()
        compact_feat_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        params_dict = {}

        # visualiza fusion module's weight
        self.vis_fusion_weight()

        # for self.vis_attention
        num_images = len(self.testloader)
        spec_index = getattr(self.cfg, 'spec_index', -1)
        if spec_index == -1:
            spec_index = np.random.randint(0, num_images, (1,))[0]

        for i, x in enumerate(self.testloader):
            seq, view, seq_type, label = x
            seq = seq.float().cuda()

            feat_full, feat_local, feat_compact, params, features = self.model(seq)
            n = feat_full.size(0)
            full_feat_list.append(feat_full.view(n, -1).data.cpu().numpy())
            local_feat_list.append(feat_local.view(n, -1).data.cpu().numpy())
            compact_feat_list.append(feat_compact.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list.append(label.item())
            params_dict[i] = params

            # visualiza the attention maps of different branches
            if i == spec_index:
                self.vis_attention(features)

        # collect localization parameters and visualize the distribution
        self.vis_loc_param_dist(params_dict)

        self.print_log('Test Full')
        acc_full = self._compute_accuracy(full_feat_list, view_list, seq_type_list,
                                          label_list)
        self.print_log('Test Local')
        acc_local = self._compute_accuracy(local_feat_list, view_list, seq_type_list,
                                           label_list)
        self.print_log('Test Compact')
        acc_compact = self._compute_accuracy(compact_feat_list, view_list, seq_type_list,
                                             label_list)

        if len(acc_compact) > 1:
            self.writer.add_scalar('test/fullAccNM', acc_full[0], self.iter)
            self.writer.add_scalar('test/fullAccBG', acc_full[1], self.iter)
            self.writer.add_scalar('test/fullAccCL', acc_full[2], self.iter)
            self.writer.add_scalar('test/localAccNM', acc_local[0], self.iter)
            self.writer.add_scalar('test/localAccBG', acc_local[1], self.iter)
            self.writer.add_scalar('test/localAccCL', acc_local[2], self.iter)
            self.writer.add_scalar('test/compactAccNM', acc_compact[0], self.iter)
            self.writer.add_scalar('test/compactAccBG', acc_compact[1], self.iter)
            self.writer.add_scalar('test/compactAccCL', acc_compact[2], self.iter)
        else:
            self.writer.add_scalar('test/fullAcc', acc_full[0], self.iter)
            self.writer.add_scalar('test/localAcc', acc_local[0], self.iter)
            self.writer.add_scalar('test/compactAcc', acc_compact[0], self.iter)
        return acc_compact


    def vis_attention(self, features):
        features = [i.cpu() for i in features]
        gl, head, torso, legs = features
        k = len(gl[0, 0]) // 2
        def f(img, name):
            img = img.mean(0).squeeze().numpy()
            fig = plt.figure()
            sns.heatmap(img)
            self.writer.add_figure('attention/{}_{}'.format(self.iter, name),
                                   fig, self.iter)
            directory = os.path.join(self.cfg.work_dir, 'figs', 'attention')
            os.makedirs(directory, exist_ok=True)
            save_path = os.path.join(directory, '{}_{}.pdf'.format(self.iter, name))
            plt.savefig(save_path)
            plt.close()
        f(gl[:, k], 'global')
        f(head[:, k], 'head')
        f(torso[:, k], 'torso')
        f(legs[:, k], 'legs')
        return


    def vis_fusion_weight(self):
        weight = self.model.module.local['feature_fusion'][0].weight.data
        y = weight.squeeze().detach().abs().mean(0).cpu().numpy()
        x = np.arange(128 + 3*64)
        assert x.shape == y.shape, '{} vs. {}'.format(x.shape, y.shape)

        fig = plt.figure()
        sns.lineplot(x=x[:128], y=y[:128], label='global')
        sns.lineplot(x=x[128:192], y=y[128:192], label='head')
        sns.lineplot(x=x[192:256], y=y[192:256], label='torso')
        sns.lineplot(x=x[256:], y=y[256:], label='legs')
        self.writer.add_figure('fusion_weight/{}'.format(self.iter),
                               fig, self.iter)
        directory = os.path.join(self.cfg.work_dir, 'figs', 'fusion_weight')
        os.makedirs(directory, exist_ok=True)
        save_path = os.path.join(directory, '{}.pdf'.format(self.iter))
        plt.savefig(save_path)
        plt.close()
        return


    def vis_loc_param_dist(self, params):
        # cuda -> cpu
        params = {k: [[j.cpu() for j in i] for i in v]
                  for k, v in params.items()}
        # save localization parameters
        if getattr(self, 'params_dict', None) is None:
            self.params_dict = {}
        self.params_dict[self.iter] = params
        path = os.path.join(self.work_dir, 'loc_params.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self.params_dict, f)
        self.print_log('Save Localization Parameters to {}'.format(path))

        # visualize parameters over all test images
        dist = defaultdict(lambda: np.array([]))
        def f(_name, _param):
            dist[_name] = np.hstack([dist[_name], _param.cpu().numpy()])
        for p in params.values():
            # dx, dy, sigma, delta, gamma, Fx, Fy = head
            head, torso, legs = p
            f('head_dt', head[0])
            f('head_dx', head[1])
            f('head_dy', head[2])
            f('head_sigma', head[3])
            f('head_delta', head[4])
            f('head_delta_t', head[5])
            f('head_gamma', head[6])

            f('torso_dt', torso[0])
            f('torso_dx', torso[1])
            f('torso_dy', torso[2])
            f('torso_sigma', torso[3])
            f('torso_delta', torso[4])
            f('torso_delta_t', torso[5])
            f('torso_gamma', torso[6])

            f('legs_dt', legs[0])
            f('legs_dx', legs[1])
            f('legs_dy', legs[2])
            f('legs_sigma', legs[3])
            f('legs_delta', legs[4])
            f('legs_delta_t', legs[5])
            f('legs_gamma', legs[6])

        for k, v in dist.items():
            self.writer.add_histogram('loc_param/{}'.format(k), v, self.iter)
        return
