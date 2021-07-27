import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .basic_blocks import SetBlock, BasicConv2d, HPM
from .gaitpart import FConv, FPFE_C, FPFE_O, HP, MCM, SeparateFc


class LocalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_h, in_w, out_h, out_w,
                 anchor_x, anchor_y, out_map_size, eps=1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_h = in_h
        self.in_w = in_w
        self.out_h = out_h
        self.out_w = out_w
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y
        self.out_map_size = out_map_size
        self.eps = eps

        self.loc_convs = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.loc_fc = nn.Sequential(
            nn.Linear(out_channels * out_map_size, 5),
            nn.BatchNorm1d(5),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """ get localization parameters """
        loc_params = self.loc_convs(x)
        loc_params = loc_params.view(loc_params.size(0), -1)
        loc_params = self.loc_fc(loc_params)
        dx, dy, log_sigma2, log_delta, log_gamma = torch.split(loc_params, 1, 1)
        dx = torch.tanh(dx) * self.out_w / 2.0 + self.anchor_x
        dy = torch.tanh(dy) * self.out_h / 2.0 + self.anchor_y
        sigma2 = torch.exp(log_sigma2)
        delta = torch.exp(log_delta)
        gamma = torch.sigmoid(log_gamma)

        """ set up transform matrix """
        grid_x_i = torch.arange(0, self.out_w).view(1, -1).float().cuda().detach()
        grid_y_i = torch.arange(0, self.out_h).view(1, -1).float().cuda().detach()
        mu_x = dx + (grid_x_i - self.out_w / 2.0) * delta
        mu_y = dy + (grid_y_i - self.out_h / 2.0) * delta

        a = torch.arange(0, self.in_w).view(1,1,-1).float().cuda().detach()
        b = torch.arange(0, self.in_h).view(1,1,-1).float().cuda().detach()
        mu_x = mu_x.view(-1, self.out_w, 1)
        mu_y = mu_y.view(-1, self.out_h, 1)
        sigma2 = sigma2.view(-1, 1, 1)
        Fx = torch.exp(-1 * torch.pow(a - mu_x, 2) / (2*sigma2))
        Fy = torch.exp(-1 * torch.pow(b - mu_y, 2) / (2*sigma2))
        # normalize, sum over H and W dims
        eps_tensor_h = self.eps * torch.ones(self.in_h).cuda().detach()
        eps_tensor_w = self.eps * torch.ones(self.in_w).cuda().detach()
        Fx = Fx / torch.max(torch.sum(Fx, 2, keepdim=True), eps_tensor_w)
        Fy = Fy / torch.max(torch.sum(Fy, 2, keepdim=True), eps_tensor_h)

        """ sampling """
        Fyv = Fy.view(Fy.size(0), 1, Fy.size(1), Fy.size(2))
        Fxv = Fx.view(Fx.size(0), 1, Fx.size(1), Fx.size(2))
        Fxt = torch.transpose(Fxv, 2, 3)
        glimpse = torch.matmul(Fyv, torch.matmul(x, Fxt))
        out = glimpse * gamma.view(-1, 1, 1, 1)

        # pad
        h_hat = out.size(2)
        w_hat = out.size(3)
        pad_t = (self.out_h - h_hat) // 2
        pad_b = self.out_h - pad_t - h_hat
        out = F.pad(out, pad=(0, 0, pad_t, pad_b))

        return out, [dx, dy, sigma2, delta, gamma, Fx, Fy]


class Local2D_Pooling(nn.Module):
    r""" 2D LocalCNN Block, integrating into GaitPart

    """
    def __init__(self, in_channels, out_channels, height, width, num_part,
                 out_map_size):
        super().__init__()
        self.height = height
        self.width = width
        self.num_part = num_part

        self.blocks = nn.ModuleList([
            LocalBlock(in_channels, out_channels, height, width,
                       height // num_part, width,
                       int(round(width / 2.0)),
                       int(round(height / num_part / 2.0 * (i+1))),
                       out_map_size)
            for i in range(num_part)
        ])

    def forward(self, x):
        N, T, C, H, W = x.size()
        x = x.view(N*T, C, H, W)
        features = []
        deltas = []
        for m in self.blocks:
            f, param = m(x)
            f = f.view(f.size(0), f.size(1), -1)
            features.append(f.mean(2) + f.max(2)[0])
            deltas.append(param[3])
        features = torch.stack(features, dim=-1)
        features = features.view(N, T, features.size(1), self.num_part)
        return features, deltas

    def local_features(self, x):
        N, T, C, H, W = x.size()
        x = x.view(N*T, C, H, W)
        features = []
        for m in self.blocks:
            f, param = m(x)
        features = torch.cat(features, dim=2)
        return features



class LocalCNN2D(nn.Module):
    def __init__(self, out_channels=256, num_part=4):
        super().__init__()
        self.layer1 = FPFE_C()
        self.layer2 = Local2D_Pooling(128, 128, 16, 11, num_part, 2*2)
        self.layer3 = MCM(channels=128, num_part=num_part, squeeze_ratio=4)
        self.layer4 = SeparateFc(num_part, 128, out_channels)


    def forward(self, silho):
        x = silho.unsqueeze(2)

        out = self.layer1(x)
        out, deltas = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out, deltas

    def local_features(self, silho):
        x = silho.unsqueeze(2)

        out = self.layer1(x)
        features = self.layer2.local_features(out)
        return features
