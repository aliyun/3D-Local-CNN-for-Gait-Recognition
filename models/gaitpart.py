import pdb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GaitPartC', 'GaitPartO']

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)

class FConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, p=1):
        super().__init__()
        self.conv = BasicConv2d(in_channels, out_channels, kernel_size,
                                padding=padding)
        self.p = p

    def forward(self, x):
        N, C, H, W = x.size()
        stripes = torch.chunk(x, self.p, dim=2)
        concated = torch.cat(stripes, dim=0)
        out = F.leaky_relu(self.conv(concated), inplace=False)
        out = torch.cat(torch.chunk(out, self.p, dim=0), dim=2)
        return out


class FPFE_C(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = FConv(1, 32, 5, 2, p=1)
        self.layer2 = FConv(32, 32, 3, 1, p=1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.layer3 = FConv(32, 64, 3, 1, p=4)
        self.layer4 = FConv(64, 64, 3, 1, p=4)

        self.layer5 = FConv(64, 128, 3, 1, p=8)
        self.layer6 = FConv(128, 128, 3, 1, p=8)

    def forward(self, x):
        N, T, C, H, W = x.size()
        out = x.view(-1, C, H, W)

        out = self.maxpool(self.layer2(self.layer1(out)))
        out = self.maxpool(self.layer4(self.layer3(out)))
        out = self.layer6(self.layer5(out))

        _, outC, outH, outW = out.size()
        out = out.view(N, T, outC, outH, outW)
        return out

class FPFE_O(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = FConv(1, 32, 5, 2, p=2)
        self.layer2 = FConv(32, 32, 3, 1, p=2)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.layer3 = FConv(32, 64, 3, 1, p=2)
        self.layer4 = FConv(64, 64, 3, 1, p=2)

        self.layer5 = FConv(64, 128, 3, 1, p=8)
        self.layer6 = FConv(128, 128, 3, 1, p=8)

        self.layer7 = FConv(128, 256, 3, 1, p=8)
        self.layer8 = FConv(256, 256, 3, 1, p=8)

    def forward(self, x):
        N, T, C, H, W = x.size()
        out = x.view(-1, C, H, W)

        out = self.maxpool(self.layer2(self.layer1(out)))
        out = self.maxpool(self.layer4(self.layer3(out)))
        out = self.layer6(self.layer5(out))
        out = self.layer8(self.layer7(out))

        _, outC, outH, outW = out.size()
        out = out.view(N, T, outC, outH, outW)
        return out


class HP(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        N, T, C, H, W = x.size()
        out = x.view(N, T, C, self.p, -1)
        out = out.mean(4) + out.max(4)[0]
        return out


class MTB1(nn.Module):
    def __init__(self, channels=128, num_part=16, squeeze_ratio=4):
        super().__init__()

        self.avgpool = nn.AvgPool1d(3, padding=1, stride=1)
        self.maxpool = nn.MaxPool1d(3, padding=1, stride=1)

        hidden_channels = channels // squeeze_ratio
        self.conv1 = nn.Conv1d(channels * num_part,
                               hidden_channels * num_part,
                               kernel_size=3, padding=1, groups=num_part)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(hidden_channels * num_part,
                               channels * num_part,
                               kernel_size=1, padding=0, groups=num_part)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, T = x.size()

        Sm = self.avgpool(x) + self.maxpool(x)
        attention = self.sigmoid(self.conv2(self.relu(self.conv1(x))))
        out = Sm * attention
        return out

class MTB2(nn.Module):
    def __init__(self, channels=128, num_part=16, squeeze_ratio=4):
        super().__init__()

        self.avgpool = nn.AvgPool1d(5, padding=2, stride=1)
        self.maxpool = nn.MaxPool1d(5, padding=2, stride=1)

        hidden_channels = channels // squeeze_ratio
        self.conv1 = nn.Conv1d(channels * num_part,
                               hidden_channels * num_part,
                               kernel_size=3, padding=1, groups=num_part)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(hidden_channels * num_part,
                               channels * num_part,
                               kernel_size=3, padding=1, groups=num_part)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, T = x.size()

        Sm = self.avgpool(x) + self.maxpool(x)
        attention = self.sigmoid(self.conv2(self.relu(self.conv1(x))))
        out = Sm * attention
        return out


class MCM(nn.Module):
    def __init__(self, channels, num_part, squeeze_ratio=4):
        super().__init__()
        self.layer1 = MTB1(channels, num_part, squeeze_ratio)
        self.layer2 = MTB2(channels, num_part, squeeze_ratio)

    def forward(self, x):
        N, T, C, M = x.size()
        out = x.permute(0, 3, 2, 1).contiguous().view(N, M*C, T)
        out = self.layer1(out) + self.layer2(out)
        out = out.max(2)[0]
        return out.view(N, M, C)


class SeparateFc(nn.Module):
    def __init__(self, num_bin, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Conv2d(num_bin * in_dim, num_bin * out_dim, kernel_size=1,
                            groups=num_bin, padding=0, bias=False)

    def forward(self, x):
        N, M, C = x.size()
        out = x.view(N, M*C, 1, 1)
        out = self.fc(out)
        return out.view(N, M, -1)


class _gaitpart(nn.Module):
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

    def forward(self, silho):
        r"""
        N: batch size
        T: num of frames
        C: channels
        H: height
        W: width
        M: num of parts
        """
        x = silho.unsqueeze(2)

        out = self.backbone(x)
        out = self.spatial_pool(out)
        out = self.temporal_pool(out)
        out = self.hpm(out)
        return out


class GaitPartC(_gaitpart):
    def __init__(self, out_channels=256):
        super().__init__()
        self.backbone = FPFE_C()
        self.spatial_pool = HP(p=16)
        self.temporal_pool = MCM(channels=128, num_part=16, squeeze_ratio=4)
        self.hpm = SeparateFc(16, 128, out_channels)

        self.reset_parameters()


class GaitPartO(_gaitpart):
    def __init__(self, out_channels=256):
        super().__init__()
        self.backbone = FPFE_O()
        self.spatial_pool = HP(p=16)
        self.temporal_pool = MCM(channels=256, num_part=16, squeeze_ratio=4)
        self.hpm = SeparateFc(16, 256, out_channels)

        self.reset_parameters()
