import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LabelSmoothCrossEntropyLoss']


class LabelSmoothCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, eps=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.confidence = 1.-eps

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(-1)
        loss = self.confidence * nll_loss + self.eps * smooth_loss
        return loss.mean()


if __name__ == '__main__':
    num_samples = 3
    num_classes = 3
    x = torch.randn(num_samples, num_classes)
    y = torch.randint(num_classes, (num_samples,))
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = LabelSmoothCrossEntropyLoss(3, eps=0)

    loss1 = criterion1(x, y)
    loss2 = criterion2(x, y)
    print('Cross Entropy Loss: {}'.format(loss1.item()))
    print('Label Smooth Cross Entropy Loss: {}'.format(loss2.item()))
