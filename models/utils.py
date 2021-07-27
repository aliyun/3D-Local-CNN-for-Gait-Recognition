import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'C3DBlock',
    'BasicConv2d',
    'FConv',
    'HP',
    'MTB1',
    'MTB2',
    'MCM',
    'SeparateFc',
]


class C3DBlock(nn.Module):
    r""" No BatchNorm or Conv bias """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               bias=False)
        self.conv2 = nn.Conv3d(out_channels,
                               out_channels,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        N, C, T, H, W = x.size()
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        return out


class CompactBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.9):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.bn2(out)
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              bias=False,
                              padding=padding,
                              **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)


class FConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, p=1):
        super().__init__()
        self.conv = BasicConv2d(in_channels,
                                out_channels,
                                kernel_size,
                                padding=padding)
        self.p = p

    def forward(self, x):
        N, C, H, W = x.size()
        stripes = torch.chunk(x, self.p, dim=2)
        concated = torch.cat(stripes, dim=0)
        out = F.leaky_relu(self.conv(concated), inplace=False)
        out = torch.cat(torch.chunk(out, self.p, dim=0), dim=2)
        return out


class HP(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        N, C, T, H, W = x.size()
        out = x.view(N, C, T, self.p, -1)
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
                               kernel_size=3,
                               padding=1,
                               groups=num_part)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(hidden_channels * num_part,
                               channels * num_part,
                               kernel_size=1,
                               padding=0,
                               groups=num_part)
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
                               kernel_size=3,
                               padding=1,
                               groups=num_part)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(hidden_channels * num_part,
                               channels * num_part,
                               kernel_size=3,
                               padding=1,
                               groups=num_part)
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
        N, C, T, M = x.size()
        out = x.permute(0, 3, 1, 2).contiguous().view(N, M * C, T)
        out = self.layer1(out) + self.layer2(out)
        out = out.max(2)[0]
        return out.view(N, M, C)


class SeparateFc(nn.Module):
    def __init__(self, num_bin, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Conv2d(num_bin * in_dim,
                            num_bin * out_dim,
                            kernel_size=1,
                            groups=num_bin,
                            padding=0,
                            bias=False)

    def forward(self, x):
        N, M, C = x.size()
        out = x.view(N, M * C, 1, 1)
        out = self.fc(out)
        return out.view(N, M, -1)
