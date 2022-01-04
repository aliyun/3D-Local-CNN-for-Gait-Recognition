#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : ZhenHuang
Date               : 2021-08-13 16:30
Last Modified By   : ZhenHuang
Last Modified Date : 2021-08-13 17:57
Description        : cfg for gifs 
-------- 
Copyright (c) 2021 Alibaba Inc. 
'''
import math
import cv2
import numpy as np

def sampling(img, bbox):
    H, W, C = img.shape
    out = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    out_h = int(1. * W * out.shape[0] / out.shape[1])
    out = cv2.resize(out, (W, out_h), interpolation=cv2.INTER_CUBIC)
    pad_t = (H - out_h) // 2
    pad_b = H - pad_t - out_h
    out = np.pad(out, ((pad_t, pad_b), (0, 0), (0, 0)))
    return out

def cfg(index):
    if index == 49:
        T = 64
        head = [(1, 9, 14, 30)] * T
        torso = [(8, 40, 2, 42)] * T
        armL = [(26, 38,
            int(14+9*math.sin(math.pi / 12 * i)),
            int(26+9*math.sin(math.pi / 12 * i)))
            for i in range(T)]
        armR = [(26, 38,
            int(16-9*math.sin(math.pi / 12 * i)),
            int(28-9*math.sin(math.pi / 12 * i)))
            for i in range(T)]
        legL = [(39, 63,
            int(12+9*math.sin(math.pi / 12 * i)),
            int(32+9*math.sin(math.pi / 12 * i)))
            for i in range(T)]
        legR = [(39, 63,
            int(12-9*math.sin(math.pi / 12 * i)),
            int(32-9*math.sin(math.pi / 12 * i)))
            for i in range(T)]
        return (head, torso, armL, armR, legL, legR)
    elif index == 54:
        T = 91
        head = [(1, 9, 14, 30)] * T
        torso = [(8, 40, 2, 42)] * T
        armL = [(26, 38, 7, 19)] * T
        armR = [(26, 38, 25, 37)] * T
        legL = [(38, 64, 3, 23)] * T
        legR = [(39, 64, 21, 41)] * T
        return (head, torso, armL, armR, legL, legR)
    elif index == 300:
        T = 105
        head = [(1, 9, 14, 30)] * T
        torso = [(8, 40, 2, 42)] * T
        armL = [(26, 38,
            int(10+6*math.sin(math.pi / 12 * i)),
            int(22+6*math.sin(math.pi / 12 * i)))
            for i in range(T)]
        armR = [(26, 38,
            int(22-4*math.sin(math.pi / 12 * i)),
            int(34-4*math.sin(math.pi / 12 * i)))
            for i in range(T)]
        legL = [(39, 63,
            int(12+9*math.sin(math.pi / 12 * i)),
            int(32+9*math.sin(math.pi / 12 * i)))
            for i in range(T)]
        legR = [(39, 63,
            int(12-9*math.sin(math.pi / 12 * i)),
            int(32-9*math.sin(math.pi / 12 * i)))
            for i in range(T)]
        return (head, torso, armL, armR, legL, legR)
    elif index == 530:
        T = 70
        head = [(1, 9, 14, 30)] * T
        torso = [(8, 40, 2, 42)] * T
        armL = [(26, 38,
            int(14+9*math.sin(math.pi / 12 * i)),
            int(26+9*math.sin(math.pi / 12 * i)))
            for i in range(T)]
        armR = [(26, 38,
            int(20-4*math.sin(math.pi / 12 * i)),
            int(33-4*math.sin(math.pi / 12 * i)))
            for i in range(T)]
        legL = [(39, 63,
            int(12+9*math.sin(math.pi / 12 * i)),
            int(32+9*math.sin(math.pi / 12 * i)))
            for i in range(T)]
        legR = [(39, 63,
            int(12-9*math.sin(math.pi / 12 * i)),
            int(32-9*math.sin(math.pi / 12 * i)))
            for i in range(T)]
        return (head, torso, armL, armR, legL, legR)