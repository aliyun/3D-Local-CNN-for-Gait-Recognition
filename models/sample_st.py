#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : ZhenHuang
Date               : 2021-08-12 09:48
Last Modified By   : ZhenHuang
Last Modified Date : 2021-08-24 21:49
Description        : Sampler with Progressive Updated offset
-------- 
Copyright (c) 2021 Alibaba Inc. 
'''

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

__all__ = [
    'GaussianSampleST',
]


def ProgressiveOffset(
    x: Tensor,
    k: int = 2,
) -> Tensor:
    N, T = x.shape
    offset = torch.cat([x.detach(), torch.zeros(N, k).cuda().float()], dim=1)
    for i in range(1, k + 1):
        x += offset[:, range(-i, T - i)] * 1. / k
    return x


class GaussianSampleST(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 out_h,
                 out_w,
                 in_h,
                 in_w,
                 dt_offset,
                 dx_offset,
                 dy_offset,
                 sigma_t_offset,
                 sigma_offset,
                 delta_t_offset,
                 delta_offset,
                 reverse=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_h = out_h
        self.out_w = out_w
        self.in_h = in_h
        self.in_w = in_w
        self.dt_offset = dt_offset
        self.dx_offset = dx_offset
        self.dy_offset = dy_offset
        self.sigma_t_offset = sigma_t_offset
        self.sigma_offset = sigma_offset
        self.delta_t_offset = delta_t_offset
        self.delta_offset = delta_offset
        self.reverse = reverse
        self.eps = 1e-8

    def forward(self, x, loc_params):
        N, C, T, H, W = x.size()
        atten_out_t = T
        atten_out_w = self.out_w
        atten_out_h = int(round(atten_out_w / self.in_w * self.in_h))
        anchor_t = T * self.dt_offset
        anchor_x = W * self.dx_offset
        anchor_y = H * self.dy_offset
        """ get localization parameters """
        # dx: [N, T]
        dx, dy, log_sigma2, log_delta, log_gamma, dt, log_delta_t, log_gamma_t = loc_params
        dx = ProgressiveOffset(dx).view(-1, 1)
        dy = ProgressiveOffset(dy).view(-1, 1)
        dx = torch.tanh(dx) * self.in_w / 2.0 + anchor_x
        dy = torch.tanh(dy) * self.in_h / 2.0 + anchor_y
        sigma2 = torch.exp(log_sigma2).view(-1, 1) * self.sigma_offset
        delta = torch.exp(log_delta).view(-1, 1) * self.delta_offset
        gamma = torch.sigmoid(log_gamma).view(-1)
        dt = torch.tanh(dt) * T / 2.0 + anchor_t
        delta_t = torch.exp(log_delta_t) * self.delta_offset
        gamma_t = torch.sigmoid(log_gamma_t)
        """ set up spatial transform matrix """
        grid_x_i = torch.arange(0,
                                atten_out_w).view(1,
                                                  -1).float().cuda().detach()
        grid_y_i = torch.arange(0,
                                atten_out_h).view(1,
                                                  -1).float().cuda().detach()
        mu_x = dx + (grid_x_i - atten_out_w / 2.0) * delta
        mu_y = dy + (grid_y_i - atten_out_h / 2.0) * delta

        a = torch.arange(0, W).view(1, 1, -1).float().cuda().detach()
        b = torch.arange(0, H).view(1, 1, -1).float().cuda().detach()
        mu_x = mu_x.view(-1, atten_out_w, 1)
        mu_y = mu_y.view(-1, atten_out_h, 1)
        sigma2 = sigma2.view(-1, 1, 1)
        Fx = torch.exp(-1 * torch.pow(a - mu_x, 2) / (2 * sigma2))
        Fy = torch.exp(-1 * torch.pow(b - mu_y, 2) / (2 * sigma2))

        # normalize, sum over H and W dims
        eps_tensor_h = self.eps * torch.ones(H).cuda().detach()
        eps_tensor_w = self.eps * torch.ones(W).cuda().detach()
        Fx = Fx / torch.max(torch.sum(Fx, 2, keepdim=True), eps_tensor_w)
        Fy = Fy / torch.max(torch.sum(Fy, 2, keepdim=True), eps_tensor_h)
        """ spatial sampling """
        Fyv = Fy.view(Fy.size(0), 1, Fy.size(1), Fy.size(2))
        Fxv = Fx.view(Fx.size(0), 1, Fx.size(1), Fx.size(2))
        Fxt = torch.transpose(Fxv, 2, 3)
        x = x.permute(0, 2, 1, 3, 4).contiguous().reshape(N*T, C, H, W)
        glimpse = torch.matmul(Fyv, torch.matmul(x, Fxt))
        if self.reverse:
            Fyt = torch.transpose(Fyv, 2, 3)
            glimpse = torch.matmul(Fyt, torch.matmul(glimpse, Fxv))
        glimpse = glimpse * gamma.view(-1, 1, 1, 1)

        """ set up temporal transform matrix """
        grid_t_i = torch.arange(0, atten_out_t).view(1, -1).float().cuda().detach()
        mu_t = dt.view(-1, 1) + (grid_t_i - atten_out_t / 2.0) * delta_t.view(-1, 1)
        mu_t = mu_t.view(-1, atten_out_t, 1)
        mu_t = mu_t / (T - 1) * 2 - 1
        mu_t = mu_t[:, :, None]
        mu_s = torch.zeros(N, atten_out_t, 1, 1).cuda().detach() - 1.
        grid = torch.cat([mu_s, mu_t], -1)
        """ temporal sampling """
        _, _c, _h, _w = glimpse.size()
        glimpse = glimpse.reshape(N, -1, _c*_h*_w).permute(0, 2, 1).contiguous()
        glimpse = F.grid_sample(glimpse[:,:,:,None], grid, align_corners=False)
        glimpse = glimpse.view(N, _c, _h, _w, -1).permute(0, 1, 4, 2, 3).contiguous()

        out = glimpse * gamma_t.view(-1, 1, 1, 1, 1)

        # pad
        x_h = out.size(3)
        pad_t = (self.out_h - x_h) // 2
        pad_b = self.out_h - pad_t - x_h
        out = F.pad(out, pad=(0, 0, pad_t, pad_b))

        return out
