import torch
import torch.nn as nn

import torch.nn.functional as F
from torch_geometric.nn.conv import ChebConv


class ChebNet(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_feats, h_feats, K=2)
        self.conv2 = ChebConv(h_feats, num_classes, K=2)
        self.dropout = 0.5

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    @torch.no_grad()
    def test_forward(self, g, x):
        row, col = g.edges()
        edge_index = torch.cat(
            [row.reshape(1, -1), col.reshape(1, -1)],
            dim=0
        )
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def forward(self, g, x, no_grad=False):
        if no_grad:
            return self.test_forward(g, x)

        row, col = g.edges()
        edge_index = torch.cat(
            [row.reshape(1, -1), col.reshape(1, -1)],
            dim=0
        )
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
