import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'GaussianSample',
    'TrilinearSample',
    'MixSample',
]


class GaussianSample(nn.Module):
    atten_t_rate = {'head': 0.4, 'torso': 0.4, 'legs': 0.4}
    atten_h_rate = {'head': 1.0 / 8, 'torso': 4.0 / 8, 'legs': 3.0 / 8}
    atten_w_rate = {'head': 4.0 / 11, 'torso': 10.0 / 11, 'legs': 10.0 / 11}
    def __init__(self, locator, extractor, param_channels, in_channels, out_channels, part,
                 out_h, out_w, fix_delta_t=True, inverse=False, eps=1e-8):
        super().__init__()
        self.param_channels = param_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.part = part
        self.out_h = out_h
        self.out_w = out_w
        self.fix_delta_t = fix_delta_t
        self.inverse = inverse
        self.eps = eps

        self.localization = locator(param_channels, param_channels //2, 7)
        self.feature_extraction = extractor(in_channels, out_channels)


    def forward(self, x, param_x):
        in_t, in_h, in_w = x.size(2), x.size(3), x.size(4)
        atten_in_t = in_t * self.atten_t_rate[self.part]
        atten_in_h = in_h * self.atten_h_rate[self.part]
        atten_in_w = in_w * self.atten_w_rate[self.part]
        atten_ratio = atten_in_w / atten_in_h
        # out_t = in_t
        atten_out_t = in_t
        atten_out_w = self.out_w
        atten_out_h = int(round(atten_out_w / atten_ratio))
        anchor_t = int(round(in_t / 2.0))
        anchor_x = int(round(in_w / 2.0))
        if self.part == 'head':
            anchor_y = int(round(atten_in_h / 2.0))
        elif self.part == 'torso':
            anchor_y = int(round(atten_in_h / 2.0 + in_h *
                                 self.atten_h_rate['head']))
        elif self.part == 'legs':
            anchor_y = int(round(in_h - atten_in_h / 2.0))
        else:
            raise ValueError('No part called: {}'.format(self.part))

        """ get localization parameters """
        dt, dx, dy, log_sigma2, log_delta_t, log_delta, log_gamma = self.localization(param_x)
        dt = torch.tanh(dt) * atten_in_t / 2.0 + anchor_t
        dx = torch.tanh(dx) * atten_in_w / 2.0 + anchor_x
        dy = torch.tanh(dy) * atten_in_h / 2.0 + anchor_y
        sigma2 = torch.exp(log_sigma2)
        if self.fix_delta_t:
            delta_t = torch.exp(log_delta_t) * self.atten_t_rate[self.part]
        else:
            delta_t = torch.exp(log_delta_t) * 12 / in_t
        delta = torch.exp(log_delta) * self.atten_w_rate[self.part]
        gamma = torch.sigmoid(log_gamma)

        """ set up transform matrix """
        grid_t_i = torch.arange(0, atten_out_t).view(1, -1).float().cuda().detach()
        grid_x_i = torch.arange(0, atten_out_w).view(1, -1).float().cuda().detach()
        grid_y_i = torch.arange(0, atten_out_h).view(1, -1).float().cuda().detach()
        mu_t = dt + (grid_t_i - atten_out_t / 2.0) * delta_t
        mu_x = dx + (grid_x_i - atten_out_w / 2.0) * delta
        mu_y = dy + (grid_y_i - atten_out_h / 2.0) * delta

        c = torch.arange(0, in_t).view(1,1,-1).float().cuda().detach()
        a = torch.arange(0, in_w).view(1,1,-1).float().cuda().detach()
        b = torch.arange(0, in_h).view(1,1,-1).float().cuda().detach()
        mu_t = mu_t.view(-1, atten_out_t, 1)
        mu_x = mu_x.view(-1, atten_out_w, 1)
        mu_y = mu_y.view(-1, atten_out_h, 1)
        sigma2 = sigma2.view(-1, 1, 1)
        Ft = torch.exp(-1 * torch.pow(c - mu_t, 2) / (2*sigma2))
        Fx = torch.exp(-1 * torch.pow(a - mu_x, 2) / (2*sigma2))
        Fy = torch.exp(-1 * torch.pow(b - mu_y, 2) / (2*sigma2))
        # normalize, sum over H and W dims
        eps_tensor_t = self.eps * torch.ones(in_t).cuda().detach()
        eps_tensor_h = self.eps * torch.ones(in_h).cuda().detach()
        eps_tensor_w = self.eps * torch.ones(in_w).cuda().detach()
        Ft = Ft / torch.max(torch.sum(Ft, 2, keepdim=True), eps_tensor_t)
        Fx = Fx / torch.max(torch.sum(Fx, 2, keepdim=True), eps_tensor_w)
        Fy = Fy / torch.max(torch.sum(Fy, 2, keepdim=True), eps_tensor_h)

        """ sampling """
        Ftv = Ft.view(Ft.size(0), 1, Ft.size(1), Ft.size(2))
        Fyv = Fy.view(Fy.size(0), 1, 1, Fy.size(1), Fy.size(2))
        Fxv = Fx.view(Fx.size(0), 1, 1, Fx.size(1), Fx.size(2))
        Fxt = torch.transpose(Fxv, 3, 4)
        glimpse = torch.matmul(Fyv, torch.matmul(x, Fxt))
        glimpse = glimpse.view(glimpse.size(0), glimpse.size(1),
                               glimpse.size(2), -1)
        glimpse = torch.matmul(Ftv, glimpse)
        glimpse = glimpse.view(glimpse.size(0), glimpse.size(1),
                               atten_out_t, atten_out_h, atten_out_w)
        if self.inverse == True:
            # TODO Fix inverse transform
            Fyt = torch.transpose(Fyv, 3, 4)
            glimpse = torch.matmul(Fyt, torch.matmul(glimpse, Fxv))
            Ftt = torch.transpose(Ftv, 2, 3)
            glimpse = glimpse.view(glimpse.size(0), glimpse.size(1),
                                   glimpse.size(2), -1)
            glimpse = torch.matmul(Ftt, glimpse)
            glimpse = glimpse.view(glimpse.size(0), glimpse.size(1),
                                   in_t, in_h, in_w)
        out = glimpse * gamma.view(-1, 1, 1, 1, 1)

        # pad
        x_h = out.size(3)
        pad_t = (self.out_h - x_h) // 2
        pad_b = self.out_h - pad_t - x_h
        out = F.pad(out, pad=(0, 0, pad_t, pad_b))

        out = self.feature_extraction(out)
        f = lambda x: x.squeeze().detach()
        params = [f(dt), f(dx), f(dy), f(sigma2), f(delta), f(delta_t), f(gamma)]
        return out, torch.stack(params, 0)


class EyeSample(GaussianSample):
    r""" No sampling, just feature_extraction """
    def forward(self, x, param_x):
        batch_size = x.size(0)
        dt = torch.zeros(batch_size).cuda()
        dx = torch.zeros(batch_size).cuda()
        dy = torch.zeros(batch_size).cuda()
        sigma2 = torch.zeros(batch_size).cuda()
        delta = torch.zeros(batch_size).cuda()
        delta_t = torch.zeros(batch_size).cuda()
        gamma = torch.zeros(batch_size).cuda()
        params = [dt, dx, dy, sigma2, delta, delta_t, gamma]

        out = self.feature_extraction(x)
        return out, torch.stack(params, 0)



class TrilinearSample(GaussianSample):
    def forward(self, x, param_x):
        in_t, in_h, in_w = x.size(2), x.size(3), x.size(4)
        atten_in_t = in_t * self.atten_t_rate[self.part]
        atten_in_h = in_h * self.atten_h_rate[self.part]
        atten_in_w = in_w * self.atten_w_rate[self.part]
        atten_ratio = atten_in_w / atten_in_h
        # out_t = in_t
        atten_out_t = in_t
        atten_out_w = self.out_w
        atten_out_h = int(round(atten_out_w / atten_ratio))
        anchor_t = int(round(in_t / 2.0))
        anchor_x = int(round(in_w / 2.0))
        if self.part == 'head':
            anchor_y = int(round(atten_in_h / 2.0))
        elif self.part == 'torso':
            anchor_y = int(round(atten_in_h / 2.0 + in_h *
                                 self.atten_h_rate['head']))
        elif self.part == 'legs':
            anchor_y = int(round(in_h - atten_in_h / 2.0))
        else:
            raise ValueError('No part called: {}'.format(self.part))

        """ get localization parameters """
        dt, dx, dy, log_sigma2, log_delta_t, log_delta, log_gamma = self.localization(param_x)
        dt = torch.tanh(dt) * atten_in_t / 2.0 + anchor_t
        dx = torch.tanh(dx) * atten_in_w / 2.0 + anchor_x
        dy = torch.tanh(dy) * atten_in_h / 2.0 + anchor_y
        sigma2 = torch.exp(log_sigma2)
        if self.fix_delta_t:
            delta_t = torch.exp(log_delta_t) * self.atten_t_rate[self.part]
        else:
            delta_t = torch.exp(log_delta_t) * 12 / in_t
        delta = torch.exp(log_delta) * self.atten_w_rate[self.part]
        gamma = torch.sigmoid(log_gamma)

        """ set up transform matrix """
        # x-w, y-h
        grid_t_i = torch.arange(0, atten_out_t).float().cuda().detach()
        grid_x_i = torch.arange(0, atten_out_w).float().cuda().detach()
        grid_y_i = torch.arange(0, atten_out_h).float().cuda().detach()
        mu_t = dt + (grid_t_i - atten_out_t / 2.0) * delta_t
        mu_x = dx + (grid_x_i - atten_out_w / 2.0) * delta
        mu_y = dy + (grid_y_i - atten_out_h / 2.0) * delta
        # normalize the grid in [-1, 1]
        # BUG
        # mu_y = mu_y / in_w * 2 - 1
        mu_t = mu_t / (in_t-1) * 2 - 1
        mu_x = mu_x / (in_w-1) * 2 - 1
        mu_y = mu_y / (in_h-1) * 2 - 1
        mu_t = mu_t[:, :, None, None, None].repeat(1, 1, atten_out_h, atten_out_w, 1)
        mu_x = mu_x[:, None, None, :, None].repeat(1, atten_out_t, atten_out_h, 1, 1)
        mu_y = mu_y[:, None, :, None, None].repeat(1, atten_out_t, 1, atten_out_w, 1)
        # BUG
        # grid = torch.cat([mu_t, mu_y, mu_x], -1)
        grid = torch.cat([mu_x, mu_y, mu_t], -1)

        """ sampling """
        glimpse = F.grid_sample(x, grid)
        out = glimpse * gamma.view(-1, 1, 1, 1, 1)

        # pad
        x_h = out.size(3)
        pad_t = (self.out_h - x_h) // 2
        pad_b = self.out_h - pad_t - x_h
        out = F.pad(out, pad=(0, 0, pad_t, pad_b))

        out = self.feature_extraction(out)
        f = lambda x: x.squeeze().detach()
        params = [f(dt), f(dx), f(dy), f(sigma2), f(delta), f(delta_t), f(gamma)]
        return out, torch.stack(params, 0)



class MixSample(GaussianSample):
    def forward(self, x, param_x):
        N, C, in_t, in_h, in_w = x.size()
        x = x.view(N, C*in_t, in_h, in_w)
        atten_in_h = in_h * self.atten_h_rate[self.part]
        atten_in_w = in_w * self.atten_w_rate[self.part]
        atten_ratio = atten_in_w / atten_in_h
        atten_out_w = self.out_w
        atten_out_h = int(round(atten_out_w / atten_ratio))
        anchor_x = int(round(in_w / 2.0))
        if self.part == 'head':
            anchor_y = int(round(atten_in_h / 2.0))
        elif self.part == 'torso':
            anchor_y = int(round(atten_in_h / 2.0 + in_h *
                                 self.atten_h_rate['head']))
        elif self.part == 'legs':
            anchor_y = int(round(in_h - atten_in_h / 2.0))
        else:
            raise ValueError('No part called: {}'.format(self.part))

        """ get localization parameters """
        dt, dx, dy, log_sigma2, log_delta_t, log_delta, log_gamma = self.localization(param_x)
        dx = torch.tanh(dx) * atten_in_w / 2.0 + anchor_x
        dy = torch.tanh(dy) * atten_in_h / 2.0 + anchor_y
        sigma2 = torch.exp(log_sigma2)
        delta = torch.exp(log_delta) * self.atten_w_rate[self.part]
        gamma = torch.sigmoid(log_gamma)

        """ set up transform matrix """
        grid_x_i = torch.arange(0, atten_out_w).view(1, -1).float().cuda().detach()
        grid_y_i = torch.arange(0, atten_out_h).view(1, -1).float().cuda().detach()
        mu_x = dx + (grid_x_i - atten_out_w / 2.0) * delta
        mu_y = dy + (grid_y_i - atten_out_h / 2.0) * delta

        a = torch.arange(0, in_w).view(1,1,-1).float().cuda().detach()
        b = torch.arange(0, in_h).view(1,1,-1).float().cuda().detach()
        mu_x = mu_x.view(-1, atten_out_w, 1)
        mu_y = mu_y.view(-1, atten_out_h, 1)
        sigma2 = sigma2.view(-1, 1, 1)
        Fx = torch.exp(-1 * torch.pow(a - mu_x, 2) / (2*sigma2))
        Fy = torch.exp(-1 * torch.pow(b - mu_y, 2) / (2*sigma2))
        # normalize, sum over H and W dims
        eps_tensor_h = self.eps * torch.ones(in_h).cuda().detach()
        eps_tensor_w = self.eps * torch.ones(in_w).cuda().detach()
        Fx = Fx / torch.max(torch.sum(Fx, 2, keepdim=True), eps_tensor_w)
        Fy = Fy / torch.max(torch.sum(Fy, 2, keepdim=True), eps_tensor_h)

        """ spatial sampling (gaussian) """
        Fyv = Fy.view(Fy.size(0), 1, Fy.size(1), Fy.size(2))
        Fxv = Fx.view(Fx.size(0), 1, Fx.size(1), Fx.size(2))
        Fxt = torch.transpose(Fxv, 2, 3)
        glimpse = torch.matmul(Fyv, torch.matmul(x, Fxt))
        if self.inverse == True:
            Fyt = torch.transpose(Fyv, 2, 3)
            glimpse = torch.matmul(Fyt, torch.matmul(glimpse, Fxv))

        # pad
        x_h = glimpse.size(2)
        pad_t = (self.out_h - x_h) // 2
        pad_b = self.out_h - pad_t - x_h
        glimpse = F.pad(glimpse, pad=(0, 0, pad_t, pad_b))

        """ temporal sampling (trilinear) """
        # reshape x
        x = glimpse.view(N, C, in_t, self.out_h*self.out_w)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(N, C*self.out_h*self.out_w, in_t, 1)
        # localization parameters
        atten_out_t = in_t
        atten_in_t = in_t * self.atten_t_rate[self.part]
        anchor_t = int(round(in_t / 2.0))
        dt = torch.tanh(dt) * atten_in_t / 2.0 + anchor_t
        # BUG
        # out_t = in_t
        if self.fix_delta_t:
            delta_t = torch.exp(log_delta_t) * self.atten_t_rate[self.part]
        else:
            delta_t = torch.exp(log_delta_t) * 12 / in_t
        # sampling
        grid_t_i = torch.arange(0, atten_out_t).float().cuda().detach()
        mu_t = dt + (grid_t_i - atten_out_t / 2.0) * delta_t
        # normalize the grid in [-1, 1]
        # BUG
        # mu_t = mu_t / atten_out_t * 2 - 1
        mu_t = mu_t / (in_t-1) * 2 - 1
        mu_t = mu_t[:, :, None, None]
        mu_s = torch.zeros(N, atten_out_t, 1, 1).cuda().detach() - 1.
        # BUG
        # grid = torch.cat([mu_t, mu_s], -1)
        grid = torch.cat([mu_s, mu_t], -1)

        glimpse = F.grid_sample(x, grid)
        glimpse = glimpse.view(N, C, self.out_h, self.out_w, in_t)
        glimpse = glimpse.permute(0, 1, 4, 2, 3).contiguous()
        out = glimpse * gamma.view(-1, 1, 1, 1, 1)

        out = self.feature_extraction(out)
        f = lambda x: x.squeeze().detach()
        params = [f(dt), f(dx), f(dy), f(sigma2), f(delta), f(delta_t), f(gamma)]
        return out, torch.stack(params, 0)