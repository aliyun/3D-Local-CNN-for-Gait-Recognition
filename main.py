#!/usr/bin/env python

import pdb
import torch
import yaml
import random
import argparse

import numpy as np
import solvers


def str2bool(v):
    ''' Convert to True/False '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    ''' Load scripts arguments '''
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='Default Configurations')
    parser.add_argument('--work-dir',
                        default='./work_dir/debug',
                        help='the work folder for storing results')
    parser.add_argument('--config',
                        default=None,
                        help='path to the configuration file')

    # processor
    parser.add_argument('--solver',
                        default='Processor',
                        type=str,
                        help='Type of Solver')
    parser.add_argument('--mode',
                        default='train',
                        help='must be train or test')

    # general config
    parser.add_argument('--seed',
                        type=int,
                        default=-1,
                        help='random seed for pytorch')
    parser.add_argument('--save-interval',
                        type=int,
                        default=30,
                        help='the interval for storing models (#iteration)')
    parser.add_argument('--name', type=str, default='e', help='log path')
    parser.add_argument('--save-name',
                        type=str,
                        default='model',
                        help='Checkpoint name')
    parser.add_argument('--print-model',
                        type=str2bool,
                        default=True,
                        help='print model architecture or not')
    parser.add_argument('--print-log',
                        type=str2bool,
                        default=True,
                        help='print logging or not')
    parser.add_argument('--test-interval',
                        type=int,
                        default=1000,
                        help='the interval for testing models (#iteration)')
    parser.add_argument('--log-interval',
                        type=int,
                        default=1000,
                        help='the interval for logging (#iteration)')
    parser.add_argument(
        '--test-before-train',
        type=str2bool,
        default=True,
        help='test the pretrained model before training or not')

    # dataset
    parser.add_argument('--dataset', default='PairLoader', type=str)
    parser.add_argument('--dataset-args', default=dict(), type=dict)

    # hyper parameters
    parser.add_argument('--start-iter',
                        type=int,
                        default=0,
                        help='start training from which epoch')
    parser.add_argument('--num-iter',
                        type=int,
                        default=1,
                        help='# of epochs for training')

    # Model
    parser.add_argument('--pretrained',
                        type=str,
                        default=None,
                        help="Path of pretrained models (not load grads)")
    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help="Path of resuming checkpoint (load grads)")
    parser.add_argument(
        '--auto-resume',
        type=str2bool,
        default=False,
        help=
        "If true, automatically resume from the latest checkpoint in the work_dir"
    )

    parser.add_argument('--model', type=str, default='', help='SetNet')
    parser.add_argument('--model-args', type=dict, default={})

    # Loss
    parser.add_argument('--loss',
                        type=str,
                        default='',
                        help='Class name of loss')
    parser.add_argument('--loss-args',
                        type=dict,
                        default={},
                        help='Args for loss')

    # optim
    parser.add_argument('--optimizer',
                        type=str,
                        default='SGD',
                        help='Type of optimizer')
    parser.add_argument('--weight-decay',
                        type=float,
                        default=0.0005,
                        help='weight decay for SGD optimizer')
    parser.add_argument('--nesterov',
                        type=str2bool,
                        default=True,
                        help='use nesterov or not')
    parser.add_argument('--betas',
                        default=(0.9, 0.999),
                        type=tuple,
                        help='Betas for Adam optimizer')
    parser.add_argument('--lr-decay', type=dict, default={})

    # multi-gpu
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--mgpu', type=str2bool, default=False)

    return parser


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    Solver = getattr(solvers, arg.solver)
    p = Solver(arg)
    p.start()
