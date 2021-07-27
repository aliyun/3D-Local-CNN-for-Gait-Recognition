import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['SetNetC', 'SetNetO']

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)


class SetBlock(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)
    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1,c,h,w))
        if self.pooling:
            x = self.pool2d(x)
        _, c, h, w = x.size()
        return x.view(n, s, c, h ,w)


class HPM(nn.Module):
    r""" Horizontal Pooling Matching """
    def __init__(self, in_dim, out_dim, bin_level_num=5):
        super(HPM, self).__init__()
        self.bin_num = [2**i for i in range(bin_level_num)]
        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform(
                    torch.zeros(sum(self.bin_num), in_dim, out_dim)))])
    def forward(self, x):
        feature = list()
        n, c, h, w = x.size()
        for num_bin in self.bin_num:
            z = x.view(n, c, num_bin, -1)
            z = z.mean(3)+z.max(3)[0]
            feature.append(z)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()
        # m, n, in_dim

        feature = feature.matmul(self.fc_bin[0])
        # [m, n, out_dim] -> [n, m, out_dim]
        # Output: batch_size x num_bin x out_dim
        return feature.permute(1, 0, 2).contiguous()


class _setnet(nn.Module):
    r"""
        用一个基类统一CASIA和OUMVLP，然后各自定义一个子类
    """
    def __init__(self, out_channels, hidden_channels=[32, 64, 128],
                 in_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.batch_frame = None

        _set_in_channels = 1
        _set_channels = [32, 64, 128]
        self.set_layer1 = SetBlock(BasicConv2d(in_channels, hidden_channels[0], 5, padding=2))
        self.set_layer2 = SetBlock(BasicConv2d(hidden_channels[0], hidden_channels[0], 3, padding=1), True)
        self.set_layer3 = SetBlock(BasicConv2d(hidden_channels[0], hidden_channels[1], 3, padding=1))
        self.set_layer4 = SetBlock(BasicConv2d(hidden_channels[1], hidden_channels[1], 3, padding=1), True)
        self.set_layer5 = SetBlock(BasicConv2d(hidden_channels[1], hidden_channels[2], 3, padding=1))
        self.set_layer6 = SetBlock(BasicConv2d(hidden_channels[2], hidden_channels[2], 3, padding=1))

        # gl_layer1 and 2 are integrated after set_layer2
        self.gl_layer1 = BasicConv2d(hidden_channels[0], hidden_channels[1], 3, padding=1)
        self.gl_layer2 = BasicConv2d(hidden_channels[1], hidden_channels[1], 3, padding=1)
        # gl_layer3 and 4 are integrated after set_layer4
        self.gl_layer3 = BasicConv2d(hidden_channels[1], hidden_channels[2], 3, padding=1)
        self.gl_layer4 = BasicConv2d(hidden_channels[2], hidden_channels[2], 3, padding=1)
        self.gl_pooling = nn.MaxPool2d(2)

        self.gl_hpm = HPM(hidden_channels[-1], out_channels)
        self.x_hpm = HPM(hidden_channels[-1], out_channels)

        self.reset_parameters()


    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal(m.weight.data, 1.0, 0.02)
                nn.init.constant(m.bias.data, 0.0)


    def frame_max(self, x):
        if self.batch_frame is None:
            return torch.max(x, 1)
        else:
            _tmp = [
                torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
                ]
            max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return max_list, arg_max_list

    def frame_median(self, x):
        if self.batch_frame is None:
            return torch.median(x, 1)
        else:
            _tmp = [
                torch.median(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
                ]
            median_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_median_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return median_list, arg_median_list

    def forward(self, silho, batch_frame=None):
        # n: batch_size, s: frame_num, k: keypoints_num, c: channel
        if batch_frame is not None:
            batch_frame = batch_frame[0].data.cpu().numpy().tolist()
            num_valid_seq = len(batch_frame)
            for i in range(len(batch_frame)):
                if batch_frame[-(i + 1)] != 0:
                    break
                else:
                    num_valid_seq -= 1
            batch_frame = batch_frame[:num_valid_seq]
            frame_sum = np.sum(batch_frame)
            if frame_sum < silho.size(1):
                silho = silho[:, :frame_sum, :, :]
            self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
        n = silho.size(0)
        x = silho.unsqueeze(2)
        del silho

        x = self.set_layer1(x)
        x = self.set_layer2(x)
        gl = self.gl_layer1(self.frame_max(x)[0])
        gl = self.gl_layer2(gl)
        gl = self.gl_pooling(gl)

        x = self.set_layer3(x)
        x = self.set_layer4(x)
        gl = self.gl_layer3(gl + self.frame_max(x)[0])
        gl = self.gl_layer4(gl)

        x = self.set_layer5(x)
        x = self.set_layer6(x)
        x = self.frame_max(x)[0]
        gl = gl + x

        gl_f = self.gl_hpm(gl)
        x_f = self.x_hpm(x)

        return torch.cat([gl_f, x_f], 1)


def SetNetC(out_channels=256):
    r""" CASIA-B """
    return _setnet(out_channels=256, hidden_channels=[32, 64, 128], in_channels=1)

def SetNetO(out_channels=256):
    r""" OUMVLP """
    return _setnet(out_channels=256, hidden_channels=[64, 128, 256], in_channels=1)
