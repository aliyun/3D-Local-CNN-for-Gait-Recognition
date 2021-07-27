#! /usr/bin/env python

import os
import cv2
import time
import imageio
import argparse
import numpy as np
from time import sleep
from warnings import warn
from scipy import misc as scisc

from multiprocessing import Pool
from multiprocessing import TimeoutError as MP_TimeoutError

import solvers


class PretreatmentC():
    r""" Data preprocessing for CASIA-B """

    def __init__(self, cfg):
        # general configuration
        self.INPUT_PATH = './data/CASIA/raw/'
        # self.OUTPUT_PATH = './data/CASIA/processed'
        self.OUTPUT_PATH = './data/CASIA/high_resolution'
        self.LOG_PATH = './exps/casia/pretreatment.log'
        self.WORKERS = 64
        # self.T_H = 64
        # self.T_W = 64
        self.T_H = 128
        self.T_W = 128


    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        with open(self.LOG_PATH, 'a') as f:
            print(str, file=f)


    def start(self):
        self.print_log('\nPretreatment Start.\n' +
                       'Input path: {}\n'.format(self.INPUT_PATH) +
                       'Output path: {}\n'.format(self.OUTPUT_PATH) +
                       'Log file: {}\n'.format(self.LOG_PATH) +
                       'Worker num: {}'.format(self.WORKERS))

        id_list = os.listdir(self.INPUT_PATH)
        id_list.sort()
        # Walk the input path
        for _id in id_list:
            seq_type = os.listdir(os.path.join(self.INPUT_PATH, _id))
            seq_type.sort()
            for _seq_type in seq_type:
                view = os.listdir(os.path.join(self.INPUT_PATH, _id, _seq_type))
                view.sort()
                for _view in view:
                    seq_info = [_id, _seq_type, _view]
                    out_dir = os.path.join(self.OUTPUT_PATH, *seq_info)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    seq_path = os.path.join(self.INPUT_PATH, *seq_info)
                    self.cut_pickle(seq_info, seq_path, out_dir)
        return


    def cut_pickle(self, seq_info, seq_path, out_dir):
        seq_name = '-'.join(seq_info)
        frame_list = os.listdir(seq_path)
        frame_list.sort()
        count_frame = 0
        for _frame_name in frame_list:
            frame_path = os.path.join(seq_path, _frame_name)
            img = cv2.imread(frame_path)[:, :, 0]
            img = self.cut_img(img, seq_info, _frame_name)
            if img is not None:
                # Save the cut img
                save_path = os.path.join(out_dir, _frame_name)
                # scisc.imsave(save_path, img)
                imageio.imwrite(save_path, img)
                count_frame += 1
        # Warn if the sequence contains less than 5 frames
        if count_frame < 5:
            self.print_log('WARNNING Seq: {}, less than 5 valid data'.format(
                '-'.join(seq_info)))

        self.print_log('LOGGING Contain {} valid frames. Saved to {}.'.format(
            count_frame, out_dir))


    def cut_img(self, img, seq_info, frame_name):
        # A silhouette contains too little white pixels
        # might be not valid for identification.
        if img.sum() <= 10000:
            self.print_log('WARNNING Seq: {}, Frame: {}, no data, {:d}'.format(
                '-'.join(seq_info), frame_name, img.sum()))
            return None
        # Get the top and bottom point
        y = img.sum(axis=1)
        y_top = (y != 0).argmax(axis=0)
        y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
        img = img[y_top:y_btm + 1, :]
        # As the height of a person is larger than the width,
        # use the height to calculate resize ratio.
        _r = img.shape[1] / img.shape[0]
        _t_w = int(self.T_H * _r)
        img = cv2.resize(img, (_t_w, self.T_H), interpolation=cv2.INTER_CUBIC)
        # Get the median of x axis and regard it as the x center of the person.
        sum_point = img.sum()
        sum_column = img.sum(axis=0).cumsum()
        x_center = -1
        for i in range(sum_column.size):
            if sum_column[i] > sum_point / 2:
                x_center = i
                break
        if x_center < 0:
            self.print_log('WARNNING Seq: {}, Frame: {}, no center'.format(
                '-'.join(seq_info), frame_name))
            return None
        h_T_W = int(self.T_W / 2)
        left = x_center - h_T_W
        right = x_center + h_T_W
        if left <= 0 or right >= img.shape[1]:
            left += h_T_W
            right += h_T_W
            _ = np.zeros((img.shape[0], h_T_W))
            img = np.concatenate([_, img, _], axis=1)
        img = img[:, left:right]
        return img.astype('uint8')


class PretreatmentO(PretreatmentC):
    r""" Data preprocessing for OUMVLP """

    def __init__(self, cfg):
        # general configuration
        self.INPUT_PATH = './data/OUMVLP/raw/'
        self.OUTPUT_PATH = './data/OUMVLP/processed/'
        self.LOG_PATH = './exps/OUMVLP/pretreatment.log'
        self.WORKERS = 62

        self.T_H = 64
        self.T_W = 64


    def start(self):
        pool = Pool(self.WORKERS)
        results = list()

        self.print_log('\nPretreatment for OUMVLP Start.\n' +
                       'Input path: {}\n'.format(self.INPUT_PATH) +
                       'Output path: {}\n'.format(self.OUTPUT_PATH) +
                       'Log file: {}\n'.format(self.LOG_PATH) +
                       'Worker num: {}'.format(self.WORKERS))

        group_list = os.listdir(self.INPUT_PATH)
        group_list.sort()
        # Walk the input path
        for _group in group_list:
            _view, _seq_type = _group[:3], _group[4:]
            id_list = os.listdir(os.path.join(self.INPUT_PATH, _group))
            id_list.sort()
            for _id in id_list:
                seq_info = [_id, _seq_type, _view]
                out_dir = os.path.join(self.OUTPUT_PATH, *seq_info)
                os.makedirs(out_dir)
                seq_path = os.path.join(self.INPUT_PATH, _group, _id)
                pool.apply_async(self.cut_pickle,
                                 args=(seq_info, seq_path, out_dir))
                sleep(0.02)

        pool.close()
        unfinish = 1
        while unfinish > 0:
            unfinish = 0
            for i, res in enumerate(results):
                try:
                    res.get(timeout=0.1)
                except Exception as e:
                    if type(e) == MP_TimeoutError:
                        unfinish += 1
                        continue
                    else:
                        print('\n\n\nERROR OCCUR: PID ##%d##, ERRORTYPE: %s\n\n\n',
                              i, type(e))
                        raise e
        pool.join()
        return
