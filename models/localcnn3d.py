import pdb
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


class LocalCNN3D(nn.Module):
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
                 sampler='trilinear',
                 extractor='c3d',
                 num_classes=73,
                 out_features=256,
                 local_channels=64,
                 local_stripes=1,
                 dropout=0.9,
                 sigma_t=0.3,
                 sigma=0.1,
                 delta_t=0.4,
                 inverse=False,
                 load_baseline=None,
                 load_top=False,
                 fusion_eye=False,
                 **kwargs):
        super().__init__()
        self.local_channels = local_channels
        backbone = self._backbones[backbone]
        locator = self._locators[locator]
        sampler = self._samplers[sampler]
        extractor = self._extractors[extractor]

        self.backbone = backbone()

        # local branches
        self.local = nn.ModuleDict({
            'head':
            sampler(extractor, 64, local_channels, 16, 11, 2, 4, 0.5, 0.5,
                    1.0 / 16, sigma_t, sigma, delta_t, 4.0 / 11, inverse),
            'torso':
            sampler(extractor, 64, local_channels, 16, 11, 8, 10, 0.5, 0.5,
                    3.0 / 8, sigma_t, sigma, delta_t, 10.0 / 11, inverse),
            'armL':
            sampler(extractor, 64, local_channels, 16, 11, 3, 3, 0.5, 0.3, 0.5,
                    sigma_t, sigma, delta_t, 3.0 / 11, inverse),
            'armR':
            sampler(extractor, 64, local_channels, 16, 11, 3, 3, 0.5, 0.7, 0.5,
                    sigma_t, sigma, delta_t, 3.0 / 11, inverse),
            'legL':
            sampler(extractor, 64, local_channels, 16, 11, 6, 5, 0.5, 0.3, 0.8,
                    sigma_t, sigma, delta_t, 5.0 / 11, inverse),
            'legR':
            sampler(extractor, 64, local_channels, 16, 11, 6, 5, 0.5, 0.7, 0.8,
                    sigma_t, sigma, delta_t, 5.0 / 11, inverse),
            'feature_fusion':
            nn.Sequential(
                nn.Conv3d(local_channels * 6 + 128,
                          128,
                          kernel_size=1,
                          bias=False),
                # nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),
            ),
            'spatial_pool':
            HP(p=local_stripes),
            'temporal_pool':
            MCM(channels=local_channels,
                num_part=6 * local_stripes,
                squeeze_ratio=2),
            'hpm':
            SeparateFc(6 * local_stripes, local_channels, out_features),
        })

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

            if fusion_eye:
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
        r""" N: batch size
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
        """ Local Branches (head, torso, armL, armR, legL, legR) """
        feat_head = self.local['head'](feat4_5d)
        feat_torso = self.local['torso'](feat4_5d)
        feat_armL = self.local['armL'](feat4_5d)
        feat_armR = self.local['armR'](feat4_5d)
        feat_legL = self.local['legL'](feat4_5d)
        feat_legR = self.local['legR'](feat4_5d)
        """ Global """
        features = [
            feat6_5d, feat_head, feat_armL, feat_armR, feat_torso, feat_legL,
            feat_legR
        ]
        gl = torch.cat(features, dim=1)
        gl = self.local['feature_fusion'](gl)
        # [N, C, T, H, W] -> [N, C, T, M]
        gl = self.spatial_pool(gl)
        # [N, C, T, M] -> [N, M, C]
        gl = self.temporal_pool(gl)
        gl = self.hpm(gl)

        # N, C, T, H, W -> N, C, T, M
        out_head = self.local['spatial_pool'](feat_head)
        out_torso = self.local['spatial_pool'](feat_torso)
        out_armL = self.local['spatial_pool'](feat_armL)
        out_armR = self.local['spatial_pool'](feat_armR)
        out_legL = self.local['spatial_pool'](feat_legL)
        out_legR = self.local['spatial_pool'](feat_legR)
        out_local = torch.cat(
            [out_head, out_torso, out_armL, out_armR, out_legL, out_legR],
            dim=-1)
        # N, C, T, M -> N, M, C
        out_local = self.local['temporal_pool'](out_local)
        out_local = self.local['hpm'](out_local)

        out_full = torch.cat([gl, out_local], dim=1)

        return gl, out_local, features

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
