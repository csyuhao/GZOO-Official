import math
import torch

import torch.nn.functional as F
from torch.nn import Parameter, Linear

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree


def C_n(n, x, lamda):
    if n == 0:
        return 1
    if n == 1:
        return x - lamda
    else:
        return (x - n - lamda + 1) * C_n(n - 1, x, lamda) - (n - 1) * lamda * C_n(n - 2, x, lamda)


class PCPropGen(MessagePassing):

    def __init__(self, conv_layer, n_poly, alpha, a, b, c, **kwargs):
        super(PCPropGen, self).__init__(aggr='add', **kwargs)
        self.alpha = alpha
        self.K = conv_layer
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()
        self.a = a
        self.b = b
        self.c = c
        self.n_poly = n_poly

    def reset_parameters(self):
        self.temp.data.fill_(self.alpha)

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = self.temp

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=x.size(self.node_dim))
        row, col = edge_index
        deg1 = degree(row, x.size(self.node_dim), dtype=x.dtype)
        deg2 = degree(col, x.size(self.node_dim), dtype=x.dtype)
        deg_inv_sqrt1 = deg1.pow(-self.c)
        deg_inv_sqrt2 = deg2.pow(-self.c)
        norm = deg_inv_sqrt1[row] * deg_inv_sqrt2[col]
        edge_index1, norm1 = add_self_loops(edge_index, -norm, fill_value=self.b, num_nodes=x.size(self.node_dim))

        x_num = range(1, self.K + 1)
        x_num = list(x_num)

        lamda_num = self.a
        out_total = x * TEMP[0]
        tmp1 = [x]
        for k in range(self.n_poly):
            x = self.propagate(edge_index1, x=-x, norm=norm1, size=None)
            tmp1.append(x)
        # CH = 1
        for i in range(len(x_num)):
            out1 = 0
            # CH = CH * TEMP[i + 1]
            for j in range(self.n_poly):
                out1 = out1 + C_n(j, x_num[i], lamda_num) / math.factorial(j) * tmp1[j]
            out_total = out_total + TEMP[i + 1] * out1
        return out_total

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K, self.temp)


class PCNet(torch.nn.Module):
    def __init__(self, in_feats, h_feats, num_classes,
                 conv_layer, n_poly, alpha, a, b, c):
        super(PCNet, self).__init__()
        self.hidden = h_feats
        self.lin1 = Linear(in_feats, self.hidden)
        self.lin2 = Linear(self.hidden, num_classes)
        self.prop1 = PCPropGen(conv_layer, n_poly, alpha, a, b, c)
        self.dropout = 0.5

    def reset_parameters(self):
        self.prop1.reset_parameters()

    @torch.no_grad()
    def test_forward(self, g, x):
        row, col = g.edges()
        edge_index = torch.cat(
            [row.reshape(1, -1), col.reshape(1, -1)],
            dim=0
        )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop1(x, edge_index)
        return x

    def forward(self, g, x, no_grad=False):
        if no_grad:
            return self.test_forward(g, x)

        row, col = g.edges()
        edge_index = torch.cat(
            [row.reshape(1, -1), col.reshape(1, -1)],
            dim=0
        )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.prop1(x, edge_index)
        return x
