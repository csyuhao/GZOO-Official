import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn.pytorch as dgl_nn

from sklearn.preprocessing import normalize


def convert_adjacency_matrix(g):
    dgl_adj_tensor = g.adj()
    values = g.adj().val
    csr_row, csr_col, csr_val_indices = g.adj().csr()
    csr_val = values[csr_val_indices]
    csr_row, csr_col, csr_val = csr_row.cpu().numpy(), csr_col.cpu().detach(), csr_val.cpu().detach()
    adj = sp.csr_matrix((csr_val, csr_col, csr_row), shape=dgl_adj_tensor.shape)
    return adj


class GNNGuard(nn.Module):

    def __init__(self, threshold: float = 0.1, add_self_loop: bool = False):
        super().__init__()
        self.threshold = threshold
        self.add_self_loop = add_self_loop

    def forward(self, g, feat=None):
        g = g.remove_self_loop()
        if feat is None:
            feat = g.ndata.get('feat', None)
            if feat is None:
                raise ValueError(
                    f"The node feature matrix is not specified, please add argument 'feat' during forward or specify `g.ndata['feat']=feat`"
                )

        row, col = g.edges()
        A = feat[row]
        B = feat[col]
        att_score = F.cosine_similarity(A, B)
        att_score[att_score < self.threshold] = 0.
        # adj = g.adjacency_matrix(scipy_fmt='csr')
        adj = convert_adjacency_matrix(g)
        row, col = row.cpu().tolist(), col.cpu().tolist()
        adj[row, col] = att_score.cpu().tolist()
        adj = normalize(adj, axis=1, norm='l1')
        att_score_norm = torch.tensor(adj[row, col]).to(feat).view(-1)

        if self.add_self_loop:
            degree = (adj != 0).sum(1).A1
            self_weight = torch.tensor(1.0 / (degree + 1)).to(feat)
            att_score_norm = torch.cat([att_score_norm, self_weight])
            g = g.add_self_loop()

        att_score_norm = att_score_norm.exp()
        g.edata['ew'] = att_score_norm

        return g

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}, add_self_loop={self.add_self_loop}"


class GCNGuard(nn.Module):
    """Graph Convolution Network (GCN) with GNNGUARD
    as in `GNNGuard: Defending Graph Neural Networks against Adversarial Attacks`
    ref: https://github.com/EdisonLeeeee/GreatX/blob/dgl/graphwar/defense/model_level/gnnguard.py

    Example
    -------
    # GCNGUARD with one hidden layer
    >>> model = GCNGuard(100, 10)
    # GCNGUARD with two hidden layers
    >>> model = GCNGuard(100, 10, hids=[32, 16])
    # GCNGUARD with two hidden layers, without activation at the first layer
    >>> model = GCNGuard(100, 10, hids=[32, 16])
    """

    def __init__(self, in_feats: int, out_feats: int, hids: list = [16],
                 dropout: float = 0.5, bn: bool = False, bias: bool = True,
                 norm: str = 'both', threshold: float = 0.1, add_self_loop: bool = True):
        r"""
        Parameters
        ----------
        in_feats : int,
            the input dimensions of model
        out_feats : int,
            the output dimensions of model
        hids : list, optional
            the number of hidden units of each hidden layer, by default [16]
        dropout : float, optional
            the dropout ratio of model, by default 0.5
        bias : bool, optional
            whether to use bias in the layers, by default True
        bn: bool, optional
            whether to use `BatchNorm1d` after the convolution layer, by default False
        norm : str, optional
            How to apply the normalizer.  Can be one of the following values:

            * ``both``, where the messages are scaled with :math:`1/c_{ji}`,
            where :math:`c_{ji}` is the product of the square root of node degrees
            (i.e.,  :math:`c_{ji} = \sqrt{|\mathcal{N}(j)|}\sqrt{|\mathcal{N}(i)|}`).

            * ``square``, where the messages are scaled with :math:`1/c_{ji}^2`, where
            :math:`c_{ji}` is defined as above.

            * ``right``, to divide the aggregated messages by each node's in-degrees,
            which is equivalent to averaging the received messages.

            * ``none``, where no normalization is applied.

            * ``left``, to divide the messages sent out from each node by its out-degrees,
            equivalent to random walk normalization.
        threshold: float, optional
            threshold in `GNNGuard` class
        add_self_loop: bool, optional
            add_self_loop in `GNNGuard` class
        """

        super().__init__()

        conv = [GNNGuard(threshold, add_self_loop)]
        for hid in hids:
            conv.append(
                dgl_nn.GraphConv(in_feats, hid, bias=bias, norm=norm, activation=None, allow_zero_in_degree=True)
            )
            if bn:
                conv.append(nn.BatchNorm1d(hid))
            conv.append(nn.ReLU())
            conv.append(nn.Dropout(dropout))
            conv.append(GNNGuard(threshold, add_self_loop))
            in_feats = hid

        # `loc=1` specifies the location of features.
        self.conv1 = nn.Sequential(*conv)
        self.conv2 = dgl_nn.GraphConv(in_feats, out_feats, bias=bias, norm=norm)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    @torch.no_grad()
    def test_forward(self, g, feat):
        g = dgl.add_self_loop(g)
        for layer in self.conv1:
            if isinstance(layer, GNNGuard):
                g = layer(g, feat)
            elif isinstance(layer, dgl_nn.GraphConv):
                feat = layer(g, feat, edge_weight=g.edata['ew'])
            else:
                feat = layer(feat)

        return self.conv2(g, feat, edge_weight=g.edata['ew'])

    def forward(self, g, feat, no_grad=False):
        if no_grad:
            return self.test_forward(g, feat)

        g = dgl.add_self_loop(g)
        for layer in self.conv1:
            if isinstance(layer, GNNGuard):
                g = layer(g, feat)
            elif isinstance(layer, dgl_nn.GraphConv):
                feat = layer(g, feat, edge_weight=g.edata['ew'])
            else:
                feat = layer(feat)

        return self.conv2(g, feat, edge_weight=g.edata['ew'])
