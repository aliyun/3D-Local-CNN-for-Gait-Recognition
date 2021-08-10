#! /usr/bin/env python
import torch
from sklearn.metrics import average_precision_score
from collections import defaultdict

__all__ = ['AverageMeter']


def mean_ap(distmat, labels, infos=None, indices=None, strict=False):
    # TODO Matrix version
    if indices is None:
        indices = torch.argsort(distmat, dim=1)
    labels = torch.tensor(labels).type_as(distmat)
    matches = (labels[indices] == labels[:, None])
    if strict:
        infos = infos.t()  # [num, 5] -> [5, num]

    aps = []
    num = len(distmat)
    for i in range(num):
        if strict:
            valid = ~((infos[0][i] == infos[0][indices[i]]) &
                      (infos[2][i] == infos[2][indices[i]]) &
                      (infos[3][i] == infos[3][indices[i]]) &
                      (infos[4][i] == infos[4][indices[i]]))
        else:
            valid = torch.ones(distmat.size(1)).byte()
            valid[0] = 0
        y_true = matches[i][valid]
        y_score = -distmat[i][indices[i]][valid]
        if not torch.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return sum(aps) / len(aps)


def cmc(distmat, labels, infos=None, topk=(1, ), indices=None, strict=False):
    """ Caculate CMC Scores
        ----
        strict: Mask out the confusing samples
    """
    # TODO Matrix version
    if indices is None:
        indices = torch.argsort(distmat, dim=1)
    labels = torch.tensor(labels).type_as(distmat)
    matches = (labels[indices] == labels[:, None])
    if strict:
        infos = infos.t()  # [num, 5] -> [5, num]

    m = defaultdict(lambda: 0)
    num = len(distmat)
    for i in range(num):
        if strict:
            valid = ~((infos[0][i] == infos[0][indices[i]]) &
                      (infos[2][i] == infos[2][indices[i]]) &
                      (infos[3][i] == infos[3][indices[i]]) &
                      (infos[4][i] == infos[4][indices[i]]))
        else:
            valid = torch.ones(distmat.size(1)).byte()
            valid[0] = 0
        index = matches[i][valid].nonzero()[0]
        for k in topk:
            if index < k:
                m[k] += 1
    return tuple([m[i] / num for i in topk])


def pairwise_distance(features):
    n = len(features)
    x = torch.cat(features)
    x = x.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
    dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
    return dist.cpu()


class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / batch_size))
    return ret
