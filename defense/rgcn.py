import torch
from torch import nn
import torch.nn.functional as F

import dgl.function as fn
from torch.nn import init


class RobustGCNConv(nn.Module):
    def __init__(self, in_feats, out_feats, bias=False, gamma=1.0, activation=None):
        super().__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight_mean = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.weight_var = nn.Parameter(torch.Tensor(in_feats, out_feats))

        if bias:
            self.bias_mean = nn.Parameter(torch.Tensor(out_feats))
            self.bias_var = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias_mean', None)
            self.register_parameter('bias_var', None)

        self._gamma = gamma
        self._activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.

        """
        init.xavier_uniform_(self.weight_mean)
        init.xavier_uniform_(self.weight_var)
        if self.bias_mean is not None:
            init.zeros_(self.bias_mean)
            init.zeros_(self.bias_var)

    def forward(self, graph, feat):
        if not isinstance(feat, tuple):
            feat = (feat, feat)

        mean = torch.matmul(feat[0], self.weight_mean)
        var = torch.matmul(feat[1], self.weight_var)

        if self.bias_mean is not None:
            mean = mean + self.bias_mean
            var = var + self.bias_var

        mean = F.relu(mean)
        var = F.relu(var)

        attention = torch.exp(-self._gamma * var)

        degs = graph.in_degrees().float().clamp(min=1)
        norm1 = torch.pow(degs, -0.5).to(mean.device).unsqueeze(1)
        norm2 = norm1.square()

        with graph.local_scope():
            graph.ndata['mean'] = mean * attention * norm1
            graph.ndata['var'] = var * attention * attention * norm2
            graph.update_all(fn.copy_u('mean', 'm'), fn.sum('m', 'mean'))
            graph.update_all(fn.copy_u('var', 'm'), fn.sum('m', 'var'))

            mean = graph.ndata.pop('mean') * norm1
            var = graph.ndata.pop('var') * norm2

            if self._activation is not None:
                mean = self._activation(mean)
                var = self._activation(var)

        return mean, var


class RobustGCN(nn.Module):
    """
    Robust Graph Convolutional Networks (`RobustGCN <https://pengcui.thumedialab.com/papers/RGCN.pdf>`__)
    """

    def __init__(self, in_features: int, out_features: int,
                 n_hids: list = [16], dropout: float = 0.5, bias: bool = True, gamma: float = 1.0):
        r"""
        Parameters
        ----------
        in_features : int,
            the input dimensions of model
        out_features : int,
            the output dimensions of model
        n_hids : list, optional
            the number of hidden units of each hidden layer, by default [16]
        acts : list, optional
            the activation function of each hidden layer, by default ['relu']
        dropout : float, optional
            the dropout ratio of model, by default 0.5
        bias : bool, optional
            whether to use bias in the layers, by default True
        gamma : float, optional
            the attention weight, by default 1.0
        """

        super().__init__()

        self.mean, self.var = None, None
        assert len(n_hids) > 0
        self.conv1 = RobustGCNConv(in_features, n_hids[0], bias=bias, activation=F.relu)

        conv2 = nn.ModuleList()
        in_features = n_hids[0]
        for hid in n_hids[1:]:
            conv2.append(RobustGCNConv(in_features, hid, bias=bias, gamma=gamma, activation=F.relu))
            in_features = hid

        conv2.append(RobustGCNConv(in_features, out_features, gamma=gamma, bias=bias))
        self.conv2 = conv2
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.conv2:
            conv.reset_parameters()

    @torch.no_grad()
    def test_forward(self, g, x):
        x = self.dropout(x)
        mean, var = self.conv1(g, x)
        self.mean, self.var = mean, var

        for conv in self.conv2:
            mean, var = self.dropout(mean), self.dropout(var)
            mean, var = conv(g, (mean, var))

        std = torch.sqrt(var + 1e-8)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)
        return z

    def forward(self, g, x, no_grad=False):
        if no_grad:
            return self.test_forward(g, x)

        x = self.dropout(x)
        mean, var = self.conv1(g, x)
        self.mean, self.var = mean, var

        for conv in self.conv2:
            mean, var = self.dropout(mean), self.dropout(var)
            mean, var = conv(g, (mean, var))

        std = torch.sqrt(var + 1e-8)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)
        return z
