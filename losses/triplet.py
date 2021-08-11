import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TripletLoss', 'HardTripletLoss, FullTripletLoss']


class TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        inputs = inputs.squeeze()
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_ap)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec


class FullTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, feature, label):
        """
        N: batch size
        M: number of independent body parts
        C: feature dimention
        """
        # [N, M, C] -> [M, N, C]
        feature = feature.permute(1, 0, 2).contiguous()
        # [M, N]
        label = label.unsqueeze(0).repeat(feature.size(0), 1)

        M, N, C = feature.size()
        # [M, N, N]
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).bool().view(-1)
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).bool().view(-1)

        # [M, N, N]
        dist = self.batch_dist(feature)
        dist = dist.view(-1)

        # non-zero full
        full_hp_dist = torch.masked_select(dist, hp_mask).view(M, N, -1, 1)
        full_hn_dist = torch.masked_select(dist, hn_mask).view(M, N, 1, -1)
        full_loss_metric = F.relu(self.margin + full_hp_dist -
                                  full_hn_dist).view(M, -1)

        full_loss_metric_sum = full_loss_metric.sum(1)
        full_loss_num = (full_loss_metric != 0).sum(1).float()

        full_loss_metric_mean = full_loss_metric_sum / full_loss_num
        full_loss_metric_mean[full_loss_num == 0] = 0

        return full_loss_metric_mean.mean(
        ), full_loss_num.mean() / full_loss_metric.numel()

    def batch_dist(self, x):
        x2 = torch.sum(x**2, 2)
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(
            1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist))
        return dist


class HardTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, feature, label):
        feature = feature.permute(1, 0, 2).contiguous()
        label = label.unsqueeze(0).repeat(feature.size(0), 1)

        # feature: [n, m, d], label: [n, m]
        n, m, d = feature.size()
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).bool().view(-1)
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).bool().view(-1)

        dist = self.batch_dist(feature)
        mean_dist = dist.mean(1).mean(1)
        dist = dist.view(-1)
        # hard
        hard_hp_dist = torch.max(
            torch.masked_select(dist, hp_mask).view(n, m, -1), 2)[0]
        hard_hn_dist = torch.min(
            torch.masked_select(dist, hn_mask).view(n, m, -1), 2)[0]
        hard_loss_metric = F.relu(self.margin + hard_hp_dist -
                                  hard_hn_dist).view(n, -1)

        # hard_loss_metric_mean = torch.mean(hard_loss_metric, 1)
        return hard_loss_metric.mean()

    def batch_dist(self, x):
        x2 = torch.sum(x**2, 2)
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(
            1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist))
        return dist
