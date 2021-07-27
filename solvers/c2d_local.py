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


class C2D_Local_Solver(BaselineSolver):

    def build_optimizer(self):
        if self.cfg.optimizer == 'SGD':
            self.optimizer_backbone = self._build_sgd(
                self.model.module.backbone,
            )
            self.optimizer_top = self._build_sgd(
                self.model.module.local,
                self.model.module.spatial_pool,
                self.model.module.temporal_pool,
                self.model.module.classifier
            )

        elif self.cfg.optimizer == 'Adam':
            self.optimizer_backbone = self._build_adam(
                self.model.module.backbone,
            )
            self.optimizer_top = self._build_adam(
                self.model.module.local,
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

            output, deltas = self.model(seq)
            triplet_loss, loss_num, ratio = self.loss(output, label)

            # delta regularization
            num = len(deltas[0])
            head_reg = torch.norm(deltas[0] - 4.0/11).div(num)
            torso_reg = torch.norm(deltas[1] - 10.0/11).div(num)
            legs_reg = torch.norm(deltas[2] - 10.0/11).div(num)
            reg_loss = self.cfg.w0*head_reg + self.cfg.w1*torso_reg + self.cfg.w2*legs_reg
            loss = triplet_loss + reg_loss

            lossMeters[0].update(triplet_loss.item())
            lossMeters[1].update(loss_num.item())
            lossMeters[2].update(ratio.item())
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
                               ' - Num: {:.2e}'.format(lossMeters[1].avg) +
                               ' - Pos/Neg: {:.2f}'.format(lossMeters[2].avg) +
                               ' - Head: {:.2f}'.format(deltaMeters[0].avg) +
                               ' - Torso: {:.2f}'.format(deltaMeters[1].avg) +
                               ' - Legs: {:.2f}'.format(deltaMeters[2].avg))
                self.writer.add_scalar('train/loss', lossMeters[0].avg, self.iter)
                self.writer.add_scalar('train/loss_num', lossMeters[1].avg, self.iter)
                self.writer.add_scalar('train/ratio', lossMeters[2].avg, self.iter)
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


    def _test(self):
        self.model.eval()

        output_list = list()
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

            output, params, features = self.model(seq)
            n, num_bin, _ = output.size()
            output_list.append(output.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list.append(label.item())
            params_dict[i] = params

            # visualiza the attention maps of different branches
            if i == spec_index:
                self.vis_attention(features)

        # collect localization parameters and visualize the distribution
        self.vis_loc_param_dist(params_dict)

        return self._compute_accuracy(output_list, view_list, seq_type_list, label_list)


    def vis_attention(self, features):
        features = [i.cpu() for i in features]
        gl, head, torso, legs = features
        k = len(gl) // 2
        def f(img, name):
            img = img.mean(0).numpy()
            fig = plt.figure()
            sns.heatmap(img)
            self.writer.add_figure('attention/{}_{}'.format(self.iter, name),
                                   fig, self.iter)
            directory = os.path.join(self.cfg.work_dir, 'figs', 'attention')
            os.makedirs(directory, exist_ok=True)
            save_path = os.path.join(directory, '{}_{}.pdf'.format(self.iter,
                                                                   name))
            plt.savefig(save_path)
            plt.close()
        f(gl[k], 'global')
        f(head[k], 'head')
        f(torso[k], 'torso')
        f(legs[k], 'legs')
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
            f('head_dx', head[0])
            f('head_dy', head[1])
            f('head_sigma', head[2])
            f('head_delta', head[3])
            f('head_gamma', head[4])

            f('torso_dx', torso[0])
            f('torso_dy', torso[1])
            f('torso_sigma', torso[2])
            f('torso_delta', torso[3])
            f('torso_gamma', torso[4])

            f('legs_dx', legs[0])
            f('legs_dy', legs[1])
            f('legs_sigma', legs[2])
            f('legs_delta', legs[3])
            f('legs_gamma', legs[4])

        for k, v in dist.items():
            self.writer.add_histogram('loc_param/{}'.format(k), v, self.iter)
        return
