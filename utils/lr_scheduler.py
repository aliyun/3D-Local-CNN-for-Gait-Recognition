#! /usr/bin/env python
import math
from bisect import bisect_right
from torch.optim import Optimizer, Adam
import numpy as np

class LossWeightDecay:
    def __init__(self, policy='Step', warmup_epoch=0, warmup_start_value=0.01,
                 **kwargs):
        self.policy = policy
        if policy == 'Step':
            self.decay = StepDecay(**kwargs)
        elif policy == 'MultiStep':
            self.decay = MultiStepDecay(**kwargs)
        elif policy == 'MultiStepAbs':
            self.decay = MultiStepAbsDecay(**kwargs)
        elif policy == 'Cosine':
            self.decay = CosineDecay(**kwargs)
        else:
            raise ValueError('{}: No such policy'.format(policy))

        self.warmup_epoch = warmup_epoch
        self.warmup_start_value = warmup_start_value

    def __getattr__(self, name):
        return getattr(self.decay, name)

    def step(self, epoch):
        lr = self.decay.get_lr(epoch)
        if epoch < self.warmup_epoch:
            lr_start = self.warmup_start_value
            lr_end = self.decay.get_lr(self.warmup_epoch)
            alpha = (lr_end - lr_start) / self.warmup_epoch
            lr = epoch * alpha + lr_start
        return lr

class LearningRate:
    def __init__(self, optimizer=None, policy="Step", warmup_epoch=0,
                 warmup_start_value=0.01, **kwargs):

        if isinstance(optimizer, list):
            for optim in optimizer:
                if not isinstance(optim, Optimizer):
                    raise TypeError('{} is not an Optimizer'.format(
                        type(optim).__name__))
        elif not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        self.policy = policy
        if policy == 'Step':
            self.decay = StepDecay(**kwargs)
        elif policy == 'MultiStep':
            self.decay = MultiStepDecay(**kwargs)
        elif policy == 'MultiStepAbs':
            self.decay = MultiStepAbsDecay(**kwargs)
        elif policy == 'Cosine':
            self.decay = CosineDecay(**kwargs)
        else:
            raise ValueError()

        self.warmup_epoch = warmup_epoch
        self.warmup_start_value = warmup_start_value

    def __getattr__(self, name):
        return getattr(self.decay, name)

    def step(self, epoch):
        # if isinstance(self.optimizer, Adam):
            # return 0.001
        lr = self.decay.get_lr(epoch)
        if epoch < self.warmup_epoch:
            lr_start = self.warmup_start_value
            lr_end = self.decay.get_lr(self.warmup_epoch)
            alpha = (lr_end - lr_start) / self.warmup_epoch
            lr = epoch * alpha + lr_start

        if isinstance(self.optimizer, list):
            for optim in self.optimizer:
                for g in optim.param_groups:
                    g['lr'] = lr
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = lr
        return lr

class StepDecay:
    def __init__(self, base=0.1, stepsize=None, gamma=0.1):
        self.base = base
        self.stepsize = stepsize
        self.gamma = gamma

    def get_lr(self, epoch):
        if self.stepsize is None or self.stepsize <= 0:
            return self.base
        return self.base * self.gamma ** (epoch // self.stepsize)

class MultiStepDecay:
    def __init__(self, base=0.1, milestones=[], gammas=0.1):
        self.base = base
        self.milestones = sorted(milestones)
        if isinstance(gammas, (int, float)):
            gammas = [gammas] * len(milestones)
        assert len(gammas) == len(milestones)
        self.gammas = gammas

    def get_lr(self, epoch):
        section = bisect_right(self.milestones, epoch)
        return self.base * np.prod(self.gammas[:section])

class MultiStepAbsDecay:
    def __init__(self, base=0.1, milestones=[], gammas=0.1):
        self.base = base
        self.milestones = sorted(milestones)
        if isinstance(gammas, (int, float)):
            gammas = [gammas] * len(milestones)
        assert len(gammas) == len(milestones)
        self.gammas = gammas

    def get_lr(self, epoch):
        section = bisect_right(self.milestones, epoch)
        periods = [self.base] + self.gammas
        return periods[section]

class CosineDecay:
    def __init__(self, base=0.1, max_epoch=0):
        self.base = base
        self.max_epoch = max_epoch

    def get_lr(self, epoch):
        if self.max_epoch <=0:
            return self.base
        theta = math.pi * epoch / self.max_epoch
        return self.base * (math.cos(theta) + 1.0) * 0.5


