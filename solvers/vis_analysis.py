#! /usr/bin/env python
import os
import pdb
import time
import yaml
import json
import random
import shutil
import pickle
import argparse
import numpy as np
from tqdm import tqdm

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import AverageMeter
from solvers import Solver

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

__all__ = ['VisAnalysis']

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


class VisAnalysis(Solver):

    def extract_features(self):
        self.build_data()
        self.build_model()

        if self.cfg.pretrained is None:
            raise ValueError('Please appoint --pretrained.')
        self.load_checkpoint(self.cfg.pretrained, optim=False)

        self.model.eval()

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        for i, x in enumerate(tqdm(self.testloader)):
            seq, view, seq_type, label = x
            # seq = np2var(seq).float()
            seq = seq.float().cuda()

            feature = self.model(seq)
            n, num_bin, _ = feature.size()
            feature_list.append(feature.data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            # label_list += label
            label_list.append(label.item())

        feature = np.concatenate(feature_list, 0)
        view = view_list
        seq_type = seq_type_list
        label = np.array(label_list)

        data = {
            'feature': feature,
            'view': view_list,
            'seq_type': seq_type,
            'label': label,
        }
        path = os.path.join(self.work_dir, self.cfg.save_name + '.pkl')
        with open(path, 'wb') as f:
            pickle.dump(data, f)

        self.print_log('Save features to {}'.format(path))
        return


    def load_features(self):
        path = self.cfg.feature_path
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        self.print_log('Load features from {}'.format(path))

        feature = data['feature']
        view = data['view']
        seq_type = data['seq_type']
        label = data['label']
        self.print_log('Feature: {}'.format(feature.shape))
        self.print_log('View: {}'.format(len(view)))
        self.print_log('Sequence type: {}'.format(len(seq_type)))
        self.print_log('ID: {}'.format(label.shape))

        return feature, view, seq_type, label

    def plot_t_SNE(self):
        feature, view, seq_type, label = self.load_features()

        num = len(label)
        feature = feature.reshape(num, -1)
        tsne = TSNE(n_components=2, init='pca')
        result = tsne.fit_transform(feature)

        # ID
        fig = plt.figure()
        plt.scatter(result[:, 0], result[:, 1], c=label, s=3.5)

        save_path = os.path.join(self.cfg.work_dir, 'id.pdf')
        plt.savefig(save_path)
        self.print_log('Saving figure to {}'.format(save_path))
        plt.close()

        # seq_type
        seq_type_set = list(set(seq_type))
        seq_type = [seq_type_set.index(i) for i in seq_type]
        fig = plt.figure()
        plt.scatter(result[:, 0], result[:, 1], c=seq_type, s=3.5)

        save_path = os.path.join(self.cfg.work_dir, 'seq_type.pdf')
        plt.savefig(save_path)
        self.print_log('Saving figure to {}'.format(save_path))
        plt.close()

        # view
        view_set = list(set(view))
        view = [view_set.index(i) for i in view]
        fig = plt.figure()
        plt.scatter(result[:, 0], result[:, 1], c=view, s=3.5)

        save_path = os.path.join(self.cfg.work_dir, 'view.pdf')
        plt.savefig(save_path)
        self.print_log('Saving figure to {}'.format(save_path))
        plt.close()

    def plot_part_t_SNE(self):
        feature, view, seq_type, label = self.load_features()

        num_sample, num_part, _ = feature.size()
        tsne = TSNE(n_components=2, init='pca')
        result = tsne.fit_transform(feature)
        # part
        fig = plt.figure()
        plt.scatter(result[:, 0], result[:, 1], c=label, s=3.5)

        save_path = os.path.join(self.cfg.work_dir, 'id.pdf')
        plt.savefig(save_path)
        self.print_log('Saving figure to {}'.format(save_path))
        plt.close()


    def plot_loc_zone(self):
        self.build_data()

        # saving dirs
        root = os.path.join(self.cfg.work_dir, 'figs')
        os.makedirs(root, exist_ok=True)

        # load saved parameters
        path = self.cfg.pretrained
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')

        num_models = len(data)
        self.print_log('Load {} Localization Parameters from {}'.format(
            num_models, path))

        # specifed images
        num_images = len(self.testloader)
        spec_index = getattr(self.cfg, 'spec_index', -1)
        if spec_index == -1:
            spec_index = np.random.randint(0, num_images, (3,))

        # save the original image
        for j in spec_index:
            path = os.path.join(root, str(j))
            os.makedirs(path, exist_ok=True)

            seq = self.testloader.dataset[j][0]
            img = seq[len(seq) // 2]
            fig = plt.figure()
            plt.imshow(img, plt.cm.gray)
            save_path = os.path.join(path, 'image.pdf')
            plt.savefig(save_path)
            self.print_log('Saving figure to {}'.format(save_path))
            plt.close()

        for i, m in data.items():
            for j in spec_index:
                self.print_log('Iteration: {}, Sequence: {}'.format(i, j))
                path = os.path.join(root, str(j))
                os.makedirs(path, exist_ok=True)

                seq = self.testloader.dataset[j][0]

                k = len(seq) // 2
                img = seq[k]
                H, W = img.shape[0], img.shape[1]

                # x-w, y-h
                # head
                num_step_x, num_step_y = 11, 6
                dx = m[j][0][0][k].item()
                dy = m[j][0][1][k].item()
                delta = m[j][0][3][k].item()
                center_x, center_y, stepsize = dx*4, dy*4, delta*4
                xmin = int(round(max(0, center_x - num_step_x * stepsize / 2.0)))
                xmax = int(round(min(W, center_x + num_step_x * stepsize / 2.0)))
                ymin = int(round(max(0, center_y - num_step_y * stepsize / 2.0)))
                ymax = int(round(min(H, center_y + num_step_y * stepsize / 2.0)))
                head_image = (img - 0.5) * 0.2 + 0.5
                head_image[ymin:ymax, xmin:xmax] = img[ymin:ymax, xmin:xmax]

                fig = plt.figure()
                plt.imshow(head_image, plt.cm.gray)
                save_path = os.path.join(path, 'head-{}.pdf'.format(i))
                plt.savefig(save_path)
                self.print_log('Saving figure to {}'.format(save_path))
                plt.close()

                # torso
                num_step_x, num_step_y = 11, 9
                dx = m[j][1][0][k].item()
                dy = m[j][1][1][k].item()
                delta = m[j][1][3][k].item()
                center_x, center_y, stepsize = dx*4, dy*4, delta*4
                xmin = int(round(max(0, center_x - num_step_x * stepsize / 2.0)))
                xmax = int(round(min(W, center_x + num_step_x * stepsize / 2.0)))
                ymin = int(round(max(0, center_y - num_step_y * stepsize / 2.0)))
                ymax = int(round(min(H, center_y + num_step_y * stepsize / 2.0)))
                torso_image = (img - 0.5) * 0.2 + 0.5
                torso_image[ymin:ymax, xmin:xmax] = img[ymin:ymax, xmin:xmax]

                fig = plt.figure()
                plt.imshow(torso_image, plt.cm.gray)
                save_path = os.path.join(path, 'torso-{}.pdf'.format(i))
                plt.savefig(save_path)
                self.print_log('Saving figure to {}'.format(save_path))
                plt.close()

                # legs
                num_step_x, num_step_y = 11, 7
                dx, dy, delta = m[j][2][0], m[j][2][1], m[j][2][3]
                dx = m[j][2][0][k].item()
                dy = m[j][2][1][k].item()
                delta = m[j][2][3][k].item()
                center_x, center_y, stepsize = dx*4, dy*4, delta*4
                xmin = int(round(max(0, center_x - num_step_x * stepsize / 2.0)))
                xmax = int(round(min(W, center_x + num_step_x * stepsize / 2.0)))
                ymin = int(round(max(0, center_y - num_step_y * stepsize / 2.0)))
                ymax = int(round(min(H, center_y + num_step_y * stepsize / 2.0)))
                legs_image = (img - 0.5) * 0.2 + 0.5
                legs_image[ymin:ymax, xmin:xmax] = img[ymin:ymax, xmin:xmax]

                fig = plt.figure()
                plt.imshow(legs_image, plt.cm.gray)
                save_path = os.path.join(path, 'legs-{}.pdf'.format(i))
                plt.savefig(save_path)
                self.print_log('Saving figure to {}'.format(save_path))
                plt.close()

