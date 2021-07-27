#! /usr/bin/env python
import os
import sys
import pdb
import time
import yaml
import random
import shutil
import argparse
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# mgpu
import torch.distributed as dist
from torch.utils.data.sampler import RandomSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from utils import import_class, LearningRate, init_seed


class Solver():
    """ Base class for all processors """

    def __init__(self, cfg):
        self.cfg = cfg
        self.work_dir = self.cfg.work_dir
        self.writer = SummaryWriter(cfg.work_dir)
        if cfg.mgpu:
            dist_backend = 'nccl'
            torch.cuda.set_device(cfg.local_rank)
            dist.init_process_group(backend=dist_backend)
        cfg.seed = init_seed(cfg.seed)


    def start(self):
        # Work Flow
        try:
            mode = getattr(self, self.cfg.mode)
        except AttributeError:
            print("ModeError: '{}' solver has no mode '{}'".format(
                self.cfg.solver, self.cfg.mode))
        return mode()

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def debug(self):
        raise NotImplementedError


    def build_data(self):
        dataset_class = '.'.join(['datasets', self.cfg.dataset])
        dataset_class = import_class(dataset_class)
        self.trainloader, self.testloader = dataset_class(**self.cfg.dataset_args)

    def _build_one_model(self, model_name, args):
        model_class = '.'.join(['models', model_name])
        model_class = import_class(model_class)
        model = model_class(**args)
        if self.cfg.mgpu:
            return DistributedDataParallel(model.cuda(),
                                           device_ids=[self.cfg.local_rank],
                                           find_unused_parameters=True)
        else:
            return nn.DataParallel(model).cuda()

    def build_model(self):
        self.model = self._build_one_model(self.cfg.model, self.cfg.model_args)

    def _build_one_loss(self, loss_name, args={}):
        loss_class = '.'.join(['losses', loss_name])
        loss_class = import_class(loss_class)
        return loss_class(**args).cuda()

    def build_loss(self):
        self.loss = self._build_one_loss(self.cfg.loss, self.cfg.loss_args)

    def _build_sgd(self, *models, nesterov=None, weight_decay=None):
        if nesterov is None:
            nesterov = self.cfg.nesterov
        if weight_decay is None:
            weight_decay = self.cfg.weight_decay
        return optim.SGD([{'params': m.parameters()} for m in models],
                         lr=0.1,
                         momentum=0.9,
                         nesterov=nesterov,
                         weight_decay=weight_decay)

    def _build_adam(self, *models, betas=None):
        if betas is None:
            betas = self.cfg.betas
        return optim.Adam([{'params': m.parameters()} for m in models],
                          lr=0.001,
                          betas=betas)

    def build_optimizer(self):
        if self.cfg.optimizer == 'SGD':
            self.optimizer = self._build_sgd(self.model)

        elif self.cfg.optimizer == 'Adam':
            self.optimizer = self._build_adam(self.model)

        else:
            raise ValueError()
        self.lr_scheduler = LearningRate(self.optimizer, **self.cfg.lr_decay)


    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        if (not self.cfg.mgpu) or self.cfg.local_rank == 0:
            print(str)
            if self.cfg.print_log:
                if not os.path.exists(self.cfg.work_dir):
                    os.makedirs(self.cfg.work_dir)
                with open('{}/log.txt'.format(self.cfg.work_dir), 'a') as f:
                    print(str, file=f)

    def print(self, str):
        return self.print_log(str, print_time=False)


    def load(self):
        if self.cfg.auto_resume:
            self.iter = self.auto_resume()
        elif self.cfg.resume:
            self.iter = self.load_checkpoint(self.cfg.resume, optim=True)
        elif self.cfg.pretrained:
            self.load_checkpoint(self.cfg.pretrained, optim=False)

    def load_checkpoint(self, filename, optim=True):
        state = torch.load(filename)
        iter = state['iteration']
        self.model.module.load_state_dict(state['model'])
        if optim:
            self.optimizer.load_state_dict(state['optimizer'])
            self.print_log('Load weights and optim from {}'.format(filename))
        else:
            self.print_log('Load only weights from {}'.format(filename))
        return iter


    def save(self):
        if (not self.cfg.mgpu) or (self.cfg.local_rank == 0):
            is_interval = (self.iter % self.cfg.save_interval == 0)

            if is_interval:
                ckpt_dir = os.path.join(self.work_dir, 'ckpt')
                if not os.path.exists(ckpt_dir):
                    os.mkdir(ckpt_dir)
                filename = self.cfg.save_name + '-' + str(self.iter) + '.pth.tar'
                ckpt_path = os.path.join(ckpt_dir, filename)
                return self.save_checkpoint(ckpt_path)
        return False

    def save_checkpoint(self, filename):
        state = {
            'iteration': self.iter,
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, filename)
        self.print_log('Save checkpoint to {}'.format(filename))
        return self.iter


    def auto_resume(self):
        ckpt_dir = os.path.join(self.work_dir, 'ckpt')
        if os.path.exists(ckpt_dir):
            ckpts = os.listdir(ckpt_dir)
            max_iter = -1
            resume_file = None
            for ckpt in ckpts:
                iter = float(ckpt.split('-')[1].split('.')[0])
                if iter > max_iter:
                    max_iter = iter
                    resume_file = ckpt
            if resume_file is not None:
                resume_file = os.path.join(ckpt_dir, resume_file)
                if os.path.exists(resume_file):
                    self.print_log('Auto resume from {}'.format(resume_file))
                    return self.load_checkpoint(resume_file, optim=True)


    def _convert_time(self, t):
        assert t > 0, 'Input must > 0, but got {}'.format(t)
        t = int(t)
        str = ''
        if t > 86400:
            days = t // 86400
            str += '{}d'.format(days)
            t -= days * 86400
        if t > 3600:
            hours = t // 3600
            str += '{}h'.format(hours)
            t -= hours * 3600
        if t > 60:
            minutes = t // 60
            str += '{}m'.format(minutes)
            t -= minutes * 60
        str += '{}s'.format(t)
        return str
