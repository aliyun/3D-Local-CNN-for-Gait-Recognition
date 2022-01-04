import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'Localization3D',
    'LocalizationST',
]


class Localization3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_params):
        super().__init__()
        self.convs = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            nn.Conv3d(out_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool3d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(out_channels * 2 * 2 * 2, num_params),
            nn.BatchNorm1d(num_params),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return torch.split(out, 1, 1)


class LocalizationST(nn.Module):
    """ Diffferent frames have different spatial localization parameters

    Parameters
    ----------
    in_channels : int
        number of the input channels
    out_channels : int
        number of the channels of the last conv
    num_parameter : int
        number of the localization parameters

    Returns
    -------
    (dx, dy, sigma, delta, gamma): [N, T]
    (dt, delta_t, gamma_t): [N, 1]
    """
    def __init__(self, in_channels, out_channels, num_params):
        super().__init__()
        self.convs = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(3),
        )

        self.fc_spatial = nn.Sequential(
            nn.Linear(out_channels * 3 * 3, 5),
            nn.BatchNorm1d(5),
        )

        self.convs_temporal = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((3, 3, 3)),
        )
        self.fc_temporal = nn.Sequential(
            nn.Linear(out_channels * 3 * 3 * 3, 3),
            nn.BatchNorm1d(3),
        )

    def forward(self, x):
        N, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().reshape(N * T, C, H, W)
        features = self.convs(x) # [N*T, C, 3, 3]

        spatial = features.view(features.size(0), -1)
        spatial = self.fc_spatial(spatial)  # [N*T, num_params]
        spatial = spatial.reshape(N, T, -1)
        dx, dy, sigma, delta, gamma = torch.split(spatial, 1, 2)

        temporal = features.reshape(N, T, -1, 3, 3).permute(0, 2, 1, 3, 4).contiguous()
        temporal = self.convs_temporal(temporal)
        temporal = temporal.view(temporal.size(0), -1)
        temporal = self.fc_temporal(temporal)
        temporal = temporal.reshape(N, -1)
        dt, delta_t, gamma_t = torch.split(temporal, 1, 1)

        return (dx.squeeze(), dy.squeeze(), sigma.squeeze(), delta.squeeze(), gamma.squeeze(), dt.squeeze(), delta_t.squeeze(), gamma_t.squeeze())