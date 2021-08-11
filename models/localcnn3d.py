import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .sample import GaussianSample, TrilinearSample, MixSample
from .localization import Localization3D
from .utils import C3DBlock, CompactBlock, BasicConv2d, HP, MCM, SeparateFc, FConv


class Backbone(nn.Module):
    r""" Basic Feature Extraction """
    def __init__(self):
        super().__init__()

        self.layer1 = BasicConv2d(1, 32, 5, 2)
        self.layer2 = BasicConv2d(32, 32, 3, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.layer3 = BasicConv2d(32, 64, 3, 1)
        self.layer4 = BasicConv2d(64, 64, 3, 1)

        self.layer5 = BasicConv2d(64, 128, 3, 1)
        self.layer6 = BasicConv2d(128, 128, 3, 1)

    def forward(self, x):
        out2 = self.maxpool(self.layer2(self.layer1(x)))
        out4 = self.maxpool(self.layer4(self.layer3(out2)))
        out6 = self.layer6(self.layer5(out4))
        return out4, out6


class FBackbone(Backbone):
    def __init__(self):
        super().__init__()

        self.layer1 = FConv(1, 32, 5, 2, p=1)
        self.layer2 = FConv(32, 32, 3, 1, p=1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.layer3 = FConv(32, 64, 3, 1, p=4)
        self.layer4 = FConv(64, 64, 3, 1, p=4)

        self.layer5 = FConv(64, 128, 3, 1, p=8)
        self.layer6 = FConv(128, 128, 3, 1, p=8)


class LocalBranch(nn.Module):
    """ Local branch for a specifed human part

    Parameters
    ----------
    human_part : str
        name of human part, within ('head', 'torso', 'armL', 'armR', 'legL', legR')
    locator : str
        name of localization module
    sampler : str
        name of sampling module
    extractor : str
        name of feature extraction module
    param_channels : int
        number of the input channels of the localization module
    in_channels : int
        number of input channels
    out_channels : int
        number of the output channels
    reverse : bool
        whether use the reverse-transform
    """
    _in_out_hw = {
        # (in_h, in_w, out_h, out_w)
        'head': (2, 4, 16, 11),
        'torso': (8, 10, 16, 11),
        'armL': (3, 3, 16, 11),
        'armR': (3, 3, 16, 11),
        'legL': (6, 5, 16, 11),
        'legR': (6, 5, 16, 11)
    }
    _offsets = {
        # (dt, dx, dy, sigma_t, sigma, delta_t, delta)
        'head': (0.5, 0.5, 1.0 / 16, 0.3, 0.1, 0.4, 4.0 / 11),
        'torso': (0.5, 0.5, 3.0 / 8, 0.3, 0.1, 0.4, 10.0 / 11),
        'armL': (0.5, 0.3, 0.5, 0.3, 0.1, 0.4, 3.0 / 11),
        'armR': (0.5, 0.7, 0.5, 0.3, 0.1, 0.4, 3.0 / 11),
        'legL': (0.5, 0.3, 0.8, 0.3, 0.1, 0.4, 5.0 / 11),
        'legR': (0.5, 0.7, 0.8, 0.3, 0.1, 0.4, 5.0 / 11),
    }

    def __init__(self,
                 human_part,
                 locator,
                 sampler,
                 extractor,
                 param_channels,
                 in_channels,
                 out_channels,
                 reverse=False):
        super().__init__()
        self.localization = locator(param_channels, param_channels // 2, 7)
        self.feature_extraction = extractor(in_channels, out_channels)

        in_h, in_w, out_h, out_w = self._in_out_hw[human_part]
        dt, dx, dy, sigma_t, sigma, delta_t, delta = self._offsets[human_part]
        self.sampler = sampler(in_channels,
                               out_channels,
                               out_h,
                               out_w,
                               in_h,
                               in_w,
                               dt,
                               dx,
                               dy,
                               sigma_t,
                               sigma,
                               delta_t,
                               delta,
                               reverse=reverse)

    def forward(self, x, param_x):
        loc_params = self.localization(param_x)
        sampled_feature = self.sampler(x, loc_params)
        out = self.feature_extraction(sampled_feature)
        return out


class LocalBlock3D(nn.Module):
    def __init__(self,
                 locator,
                 sampler,
                 extractor,
                 param_channels,
                 in_channels,
                 local_channels,
                 out_channels,
                 reverse=False):
        super().__init__()
        self.head = LocalBranch('head', locator, sampler, extractor,
                                param_channels, in_channels, local_channels,
                                reverse)
        self.torso = LocalBranch('torso', locator, sampler, extractor,
                                 param_channels, in_channels, local_channels,
                                 reverse)
        self.armL = LocalBranch('armL', locator, sampler, extractor,
                                param_channels, in_channels, local_channels,
                                reverse)
        self.armR = LocalBranch('armR', locator, sampler, extractor,
                                param_channels, in_channels, local_channels,
                                reverse)
        self.legL = LocalBranch('legL', locator, sampler, extractor,
                                param_channels, in_channels, local_channels,
                                reverse)
        self.legR = LocalBranch('legR', locator, sampler, extractor,
                                param_channels, in_channels, local_channels,
                                reverse)

        self.feature_fusion = nn.Sequential(
            nn.Conv3d(local_channels * 6 + out_channels,
                      out_channels,
                      kernel_size=1,
                      bias=False),
            # nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, param_x):
        head = self.head(x, param_x.detach())
        torso = self.torso(x, param_x.detach())
        armL = self.armL(x, param_x.detach())
        armR = self.armR(x, param_x.detach())
        legL = self.legL(x, param_x.detach())
        legR = self.legR(x, param_x.detach())
        out = torch.cat([param_x, head, torso, armL, armR, legL, legR], dim=1)
        out = self.feature_fusion(out)
        return out


class LocalCNN3D(nn.Module):
    """ 3D Local Convolutional Neural Networks for Gait Recognition

    Parameters
    ----------
    backbone : str
        type of backbone. 'basic' is regular Conv2d, 'fconv' is the backbone from GaitPart.
    locator : str
        type of localization module
    sampler : str
        type of sampler. 'gaussian', 'trilinear' or 'mixture'.
    extractor : str
        type of feature extraction module in 3DLocalBlock.
    num_classes : int
        number of classes/identities for softmax loss training.
    out_features : int
        length of output features
    fs_weight_init_eye : bool
        whether the feature fusion module's weights are initialized with eye-matrix
    

    Returns
    -------
    [type]
        [description]
    """
    _backbones = {'basic': Backbone, 'fconv': FBackbone}
    _locators = {'3d': Localization3D}
    _samplers = {
        'gaussian': GaussianSample,
        'trilinear': TrilinearSample,
        'mix': MixSample
    }
    _extractors = {'c3d': C3DBlock}

    def __init__(self,
                 backbone='basic',
                 locator='3d',
                 sampler='mix',
                 extractor='c3d',
                 num_classes=73,
                 out_features=256,
                 local_channels=64,
                 local_stripes=1,
                 dropout=0.9,
                 reverse=False,
                 load_baseline=None,
                 load_top=False,
                 fs_weight_init_eye=False,
                 **kwargs):
        super().__init__()
        self.local_channels = local_channels
        backbone = self._backbones[backbone]
        locator = self._locators[locator]
        sampler = self._samplers[sampler]
        extractor = self._extractors[extractor]

        self.backbone = backbone()

        # local branches
        self.local = LocalBlock3D(locator,
                                  sampler,
                                  extractor,
                                  128,
                                  64,
                                  local_channels,
                                  128,
                                  reverse=reverse)

        self.spatial_pool = HP(p=16)
        self.temporal_pool = MCM(channels=128, num_part=16, squeeze_ratio=4)
        self.hpm = SeparateFc(16, 128, out_features)

        self.reset_parameters()

        if load_baseline:
            state_dict = torch.load(load_baseline)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self.load_from_baseline(state_dict, load_top)
            print('Load baseline weights from {}'.format(load_baseline))

            if fs_weight_init_eye:
                # make sure backbone.feature_fusion weights is initialized well
                weight = self.local['feature_fusion'][0].weight
                out_c = weight.shape[0]
                new = 1e-8 * torch.ones_like(weight)
                new[:out_c, :out_c, 0, 0, 0] = torch.eye(out_c)
                weight.data.copy_(new)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m,
                            (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, silho):
        """ N: batch size
             T: num of frames
             C: channels
             H: height
             W: width
             M: num of parts
        """
        x = silho.unsqueeze(2)
        """ Backbone """
        N, T, C, H, W = x.size()
        x = x.view(-1, C, H, W)
        feat4, feat6 = self.backbone(x)
        # feat4: N*T, C, H, W -> N, C, T, H, W
        feat4_5d = feat4.view(N, T,
                              *feat4.size()[1:]).permute(0, 2, 1, 3,
                                                         4).contiguous()
        feat6_5d = feat6.view(N, T,
                              *feat6.size()[1:]).permute(0, 2, 1, 3,
                                                         4).contiguous()
        gl = self.local(feat4_5d, feat6_5d)
        # [N, C, T, H, W] -> [N, C, T, M]
        gl = self.spatial_pool(gl)
        # [N, C, T, M] -> [N, M, C]
        gl = self.temporal_pool(gl)
        gl = self.hpm(gl)

        return gl

    def load_from_baseline(self, state_dict, load_top=False):
        for name, param in self.named_parameters():
            if 'local' not in name:
                # baseline has no local parameters
                if load_top:
                    param.data.copy_(state_dict[name])
                elif ('backbone' in name or 'spatial_pool' in name
                      or 'temporal_pool' in name or 'hpm' in name):
                    # only load the backbone
                    param.data.copy_(state_dict[name])
        return
