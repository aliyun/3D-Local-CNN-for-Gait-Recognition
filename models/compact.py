import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'CompactBlock',
    'CompactBlockB',
    'CompactBlockC',
    'CompactBlockPro',
    'CompactBlock_DropA',
    'CompactBlock_DropB',
]

class CompactBlock_DropA(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.9):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout)
        self.fc = nn.Linear(in_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

    def forward(self, x):
        N, M, C = x.size()
        out = self.bn1(x.view(N, -1))
        out = self.relu(out)
        out = self.dropout(out.view(N, M, C))
        out = self.fc(out.view(N, -1))
        out = self.bn2(out)
        return out

class CompactBlock_DropB(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.9):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout)
        self.fc = nn.Linear(in_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

    def forward(self, x):
        N, M, C = x.size()
        out = self.bn1(x.view(N, -1))
        out = self.relu(out)
        out = self.dropout(out.view(N, M, C).permute(0, 2, 1).contiguous())
        out = self.fc(out.view(N, -1))
        out = self.bn2(out)
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

class CompactBlockB(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.9):
        super().__init__()
        # self.bn1 = nn.BatchNorm1d(in_features)
        # self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

    def forward(self, x):
        # out = self.bn1(x)
        # out = self.relu(out)
        # out = self.dropout(out)
        out = self.dropout(x)
        out = self.fc(out)
        out = self.bn2(out)
        return out

class CompactBlockC(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.9):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, out_features)
        # self.bn2 = nn.BatchNorm1d(out_features)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        # out = self.bn2(out)
        return out

class CompactBlockPro(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,
                 dropout1=0.5, dropout2=0.5):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout1)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn2 = nn.BatchNorm1d(hidden_features)
        self.dropout2 = nn.Dropout(p=dropout2)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn3 = nn.BatchNorm1d(out_features)

    def forward(self, x):
        out = self.relu(self.bn1(x))
        out = self.dropout1(out)
        out = self.relu(self.bn2(self.fc1(out)))
        out = self.dropout2(out)
        out = self.bn3(self.fc2(out))
        return out
