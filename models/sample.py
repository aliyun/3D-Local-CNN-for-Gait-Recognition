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
        dt, dx, dy, log_sigma2, log_delta_t, log_delta, log_gamma = loc_params
        dt = torch.tanh(dt) * T / 2.0 + anchor_t
        dx = torch.tanh(dx) * self.in_w / 2.0 + anchor_x
        dy = torch.tanh(dy) * self.in_h / 2.0 + anchor_y
        sigma2 = torch.exp(log_sigma2) * self.sigma_offset
        delta_t = torch.exp(log_delta_t) * self.delta_t_offset
        delta = torch.exp(log_delta) * self.delta_offset
        gamma = torch.sigmoid(log_gamma)
        """ set up transform matrix """
        grid_t_i = torch.arange(0,
                                atten_out_t).view(1,
                                                  -1).float().cuda().detach()
        grid_x_i = torch.arange(0,
                                atten_out_w).view(1,
                                                  -1).float().cuda().detach()
        grid_y_i = torch.arange(0,
                                atten_out_h).view(1,
                                                  -1).float().cuda().detach()
        mu_t = dt + (grid_t_i - atten_out_t / 2.0) * delta_t
        mu_x = dx + (grid_x_i - atten_out_w / 2.0) * delta
        mu_y = dy + (grid_y_i - atten_out_h / 2.0) * delta

        c = torch.arange(0, T).view(1, 1, -1).float().cuda().detach()
        a = torch.arange(0, W).view(1, 1, -1).float().cuda().detach()
        b = torch.arange(0, H).view(1, 1, -1).float().cuda().detach()
        mu_t = mu_t.view(-1, atten_out_t, 1)
        mu_x = mu_x.view(-1, atten_out_w, 1)
        mu_y = mu_y.view(-1, atten_out_h, 1)
        sigma2 = sigma2.view(-1, 1, 1)
        Ft = torch.exp(-1 * torch.pow(c - mu_t, 2) / (2 * sigma2))
        Fx = torch.exp(-1 * torch.pow(a - mu_x, 2) / (2 * sigma2))
        Fy = torch.exp(-1 * torch.pow(b - mu_y, 2) / (2 * sigma2))
        # normalize, sum over H and W dims
        eps_tensor_t = self.eps * torch.ones(T).cuda().detach()
        eps_tensor_h = self.eps * torch.ones(H).cuda().detach()
        eps_tensor_w = self.eps * torch.ones(W).cuda().detach()
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
        glimpse = glimpse.view(glimpse.size(0), glimpse.size(1), atten_out_t,
                               atten_out_h, atten_out_w)
        if self.reverse:
            Fyt = torch.transpose(Fyv, 3, 4)
            glimpse = torch.matmul(Fyt, torch.matmul(glimpse, Fxv))
            Ftt = torch.transpose(Ftv, 2, 3)
            glimpse = glimpse.view(glimpse.size(0), glimpse.size(1),
                                   glimpse.size(2), -1)
            glimpse = torch.matmul(Ftt, glimpse)
            glimpse = glimpse.view(glimpse.size(0), glimpse.size(1), T, H, W)
        out = glimpse * gamma.view(-1, 1, 1, 1, 1)

        # pad
        x_h = out.size(3)
        pad_t = (self.out_h - x_h) // 2
        pad_b = self.out_h - pad_t - x_h
        out = F.pad(out, pad=(0, 0, pad_t, pad_b))

        return out


class IdentitySample(GaussianSample):
    r""" No sampling, output = input """
    def forward(self, x, param_x):
        return x


class TrilinearSample(GaussianSample):
    def forward(self, x, loc_params):
        N, C, T, H, W = x.size()
        atten_out_t = T
        atten_out_w = self.out_w
        atten_out_h = int(round(atten_out_w / self.in_w * self.in_h))
        anchor_t = T * self.dt_offset
        anchor_x = W * self.dx_offset
        anchor_y = H * self.dy_offset
        """ get localization parameters """
        dt, dx, dy, log_sigma2, log_delta_t, log_delta, log_gamma = loc_params
        dt = torch.tanh(dt) * T / 2.0 + anchor_t
        dx = torch.tanh(dx) * self.in_w / 2.0 + anchor_x
        dy = torch.tanh(dy) * self.in_h / 2.0 + anchor_y
        sigma2 = torch.exp(log_sigma2) * self.sigma_offset
        delta_t = torch.exp(log_delta_t) * self.delta_t_offset
        delta = torch.exp(log_delta) * self.delta_offset
        gamma = torch.sigmoid(log_gamma)
        """ set up transform matrix """
        # x-w, y-h
        grid_t_i = torch.arange(0,
                                atten_out_t).view(1,
                                                  -1).float().cuda().detach()
        grid_x_i = torch.arange(0,
                                atten_out_w).view(1,
                                                  -1).float().cuda().detach()
        grid_y_i = torch.arange(0,
                                atten_out_h).view(1,
                                                  -1).float().cuda().detach()
        mu_t = dt + (grid_t_i - atten_out_t / 2.0) * delta_t
        mu_x = dx + (grid_x_i - atten_out_w / 2.0) * delta
        mu_y = dy + (grid_y_i - atten_out_h / 2.0) * delta
        # normalize the grid in [-1, 1]
        # mu_y = mu_y / in_w * 2 - 1
        mu_t = mu_t / (T - 1) * 2 - 1
        mu_x = mu_x / (W - 1) * 2 - 1
        mu_y = mu_y / (H - 1) * 2 - 1
        mu_t = mu_t[:, :, None, None, None].repeat(1, 1, atten_out_h,
                                                   atten_out_w, 1)
        mu_x = mu_x[:, None, None, :, None].repeat(1, atten_out_t, atten_out_h,
                                                   1, 1)
        mu_y = mu_y[:, None, :, None, None].repeat(1, atten_out_t, 1,
                                                   atten_out_w, 1)
        # BUG
        # grid = torch.cat([mu_t, mu_y, mu_x], -1)
        grid = torch.cat([mu_x, mu_y, mu_t], -1)
        """ sampling """
        glimpse = F.grid_sample(x, grid, align_corners=False)
        out = glimpse * gamma.view(-1, 1, 1, 1, 1)

        # pad
        x_h = out.size(3)
        pad_t = (self.out_h - x_h) // 2
        pad_b = self.out_h - pad_t - x_h
        out = F.pad(out, pad=(0, 0, pad_t, pad_b))

        return out


class MixSample(GaussianSample):
    def forward(self, x, loc_params):
        N, C, T, H, W = x.size()
        atten_out_t = T
        atten_out_w = self.out_w
        atten_out_h = int(round(atten_out_w / self.in_w * self.in_h))
        anchor_t = T * self.dt_offset
        anchor_x = W * self.dx_offset
        anchor_y = H * self.dy_offset
        """ get localization parameters """
        dt, dx, dy, log_sigma2, log_delta_t, log_delta, log_gamma = loc_params
        dt = torch.tanh(dt) * T / 2.0 + anchor_t
        dx = torch.tanh(dx) * self.in_w / 2.0 + anchor_x
        dy = torch.tanh(dy) * self.in_h / 2.0 + anchor_y
        sigma2 = torch.exp(log_sigma2) * self.sigma_offset
        delta_t = torch.exp(log_delta_t) * self.delta_t_offset
        delta = torch.exp(log_delta) * self.delta_offset
        gamma = torch.sigmoid(log_gamma)
        """ set up transform matrix """
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
        """ spatial sampling (gaussian) """
        Fyv = Fy.view(Fy.size(0), 1, Fy.size(1), Fy.size(2))
        Fxv = Fx.view(Fx.size(0), 1, Fx.size(1), Fx.size(2))
        Fxt = torch.transpose(Fxv, 2, 3)
        glimpse = torch.matmul(Fyv, torch.matmul(x, Fxt))
        if self.reverse == True:
            Fyt = torch.transpose(Fyv, 2, 3)
            glimpse = torch.matmul(Fyt, torch.matmul(glimpse, Fxv))

        # pad
        x_h = glimpse.size(3)
        pad_t = (self.out_h - x_h) // 2
        pad_b = self.out_h - pad_t - x_h
        glimpse = F.pad(glimpse, pad=(0, 0, pad_t, pad_b))
        """ temporal sampling (trilinear) """
        # reshape x
        x = glimpse.view(N, C, T, self.out_h * self.out_w)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(N, C * self.out_h * self.out_w, T, 1)
        # localization parameters
        atten_out_t = T
        atten_in_t = T
        anchor_t = int(round(T / 2.0))
        dt = torch.tanh(dt) * atten_in_t / 2.0 + anchor_t
        delta_t = torch.exp(log_delta_t) * self.delta_t_offset
        # sampling
        grid_t_i = torch.arange(0, atten_out_t).float().cuda().detach()
        mu_t = dt + (grid_t_i - atten_out_t / 2.0) * delta_t
        # normalize the grid in [-1, 1]
        # BUG
        # mu_t = mu_t / atten_out_t * 2 - 1
        mu_t = mu_t / (T - 1) * 2 - 1
        mu_t = mu_t[:, :, None, None]
        mu_s = torch.zeros(N, atten_out_t, 1, 1).cuda().detach() - 1.
        # BUG
        # grid = torch.cat([mu_t, mu_s], -1)
        grid = torch.cat([mu_s, mu_t], -1)

        glimpse = F.grid_sample(x, grid, align_corners=False)
        glimpse = glimpse.view(N, C, self.out_h, self.out_w, T)
        glimpse = glimpse.permute(0, 1, 4, 2, 3).contiguous()
        out = glimpse * gamma.view(-1, 1, 1, 1, 1)

        return out