import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch_geometric.nn.conv import GCNConv

import torch_sparse
from torch_sparse import coalesce
from torch_geometric.utils import dropout_edge


def linear_to_triu_idx(n: int, lin_idx: torch.Tensor) -> torch.Tensor:
    row_idx = (n - 2 - torch.floor(torch.sqrt(-8 * lin_idx.double() + 4 * n * (n - 1) - 7) / 2.0 - 0.5)).long()
    col_idx = lin_idx + row_idx + 1 - n * (n - 1) // 2 + (n - row_idx) * ((n - row_idx) - 1) // 2
    return torch.stack((row_idx, col_idx))


def inner(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1, 1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1, 1) + 1e-15)
    return (1 - (t1 * t2).sum(1)).mean()


def to_symmetric(edge_index, edge_weight, n, op='mean'):
    symmetric_edge_index = torch.cat(
        (edge_index, edge_index.flip(0)), dim=-1
    )

    symmetric_edge_weight = edge_weight.repeat(2)
    symmetric_edge_index, symmetric_edge_weight = coalesce(
        symmetric_edge_index,
        symmetric_edge_weight,
        m=n,
        n=n,
        op=op
    )
    return symmetric_edge_index, symmetric_edge_weight


def grad_with_checkpoint(outputs, inputs):
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    for input in inputs:
        if not input.is_leaf:
            input.retain_grad()
    torch.autograd.backward(outputs)

    grad_outputs = []
    for input in inputs:
        grad_outputs.append(input.grad.clone())
        input.grad.zero_()
    return grad_outputs


def bisection(edge_weights, a, b, n_perturbations, epsilon=1e-5, iter_max=1e5):
    def func(x):
        return torch.clamp(edge_weights - x, 0, 1).sum() - n_perturbations

    miu = a
    for i in range(int(iter_max)):
        miu = (a + b) / 2
        # Check if middle point is root
        if func(miu) == 0.0:
            break
        # Decide the side to repeat the steps
        if func(miu) * func(a) < 0:
            b = miu
        else:
            a = miu
        if (b - a) <= epsilon:
            break
    return miu


class OptimizeFeatureAndEdge(object):

    def __init__(self, num_nodes, num_features, epochs, loop_feat, loop_adj,
                 ratio=0.1, lr_feat=0.01, lr_adj=0.01, device='cuda'):
        self.ratio = ratio
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.eps = 1e-7
        self.lr_adj, self.lr_feat = lr_adj, lr_feat
        self.device = device
        self.epochs, self.loop_feat, self.loop_adj = epochs, loop_feat, loop_adj

    def __call__(self, model, feat, edge_index, edge_weight, train_mask, labels):
        model.eval()

        n_perturbations = int(self.ratio * edge_index.shape[1] // 2)
        modified_edge_index, perturbed_edge_weight = OptimizeFeatureAndEdge.sample_random_block(self.num_nodes, self.eps, edge_index)
        perturbed_edge_weight.requires_grad = True
        optimizer_adj = torch.optim.Adam([perturbed_edge_weight], lr=self.lr_adj)

        delta_feat = Parameter(torch.FloatTensor(self.num_nodes, self.num_features).to(self.device))
        delta_feat.data.fill_(0)
        optimizer_feat = torch.optim.Adam([delta_feat], lr=self.lr_feat)

        for it in range(self.epochs // (self.loop_adj + self.loop_feat)):
            delta_feat.requires_grad = True
            for j in range(self.loop_feat):
                optimizer_feat.zero_grad()
                cur_feat = feat + delta_feat
                loss = self.test_time_loss(model, cur_feat, edge_index, train_mask, labels, edge_weight)

                loss.backward()

                optimizer_feat.step()
                delta_feat.data = torch.clamp(delta_feat.data, min=-1., max=1.)

            n_feat = (feat + delta_feat).detach()
            for j in range(self.loop_adj):
                perturbed_edge_weight.requires_grad = True

                edge_index, edge_weight = self.get_modified_adj(edge_index, edge_weight, modified_edge_index, perturbed_edge_weight, self.num_nodes)
                loss = self.test_time_loss(model, n_feat, edge_index, train_mask, labels, edge_weight)
                gradient = grad_with_checkpoint(loss, perturbed_edge_weight)[0]
                gradient = torch.where(torch.isnan(gradient), self.eps, gradient)

                with torch.no_grad():
                    optimizer_adj.zero_grad()
                    perturbed_edge_weight.grad = gradient
                    optimizer_adj.step()
                    perturbed_edge_weight.data[perturbed_edge_weight < self.eps] = self.eps
                    perturbed_edge_weight = self.project(n_perturbations, perturbed_edge_weight, self.eps)
                    edge_index, edge_weight = self.get_modified_adj(edge_index, edge_weight, modified_edge_index, perturbed_edge_weight, self.num_nodes)

        n_feat = feat + delta_feat
        return n_feat.detach(), edge_index.detach(), edge_weight.detach()

    def project(self, n_perturbations, values, eps, inplace=False, bisec=True):
        if not inplace:
            values = values.clone()

        if not bisec:
            values.data.copy_(torch.clamp(
                values, min=eps, max=1 - eps
            ))
            return values

        if torch.clamp(values, 0, 1).sum() > n_perturbations:
            left = (values - 1).min()
            right = values.max()
            miu = bisection(values, left, right, n_perturbations)
            values.data.copy_(torch.clamp(
                values - miu, min=eps, max=1 - eps
            ))
        else:
            values.data.copy_(torch.clamp(
                values, min=eps, max=1 - eps
            ))
        return values

    def get_modified_adj(self, edge_index, edge_weight, modified_edge_index, perturbed_edge_weight, num_nodes):
        modified_edge_index, modified_edge_weight = to_symmetric(
            modified_edge_index, perturbed_edge_weight, num_nodes
        )
        edge_index = torch.cat((edge_index.to(self.device), modified_edge_index), dim=-1)
        edge_weight = torch.cat((edge_weight.to(self.device), modified_edge_weight))

        edge_index, edge_weight = torch_sparse.coalesce(edge_index, edge_weight, m=num_nodes, n=num_nodes, op='sum')

        # Allow removal of edges
        edge_weight[edge_weight >= 1] = 2 - edge_weight[edge_weight >= 1]
        return edge_index, edge_weight

    def test_time_loss(self, model, feat, edge_index, train_mask, labels, edge_weight=None):
        output = model(feat, edge_index, edge_weight)
        loss = F.cross_entropy(output[train_mask], labels[train_mask])

        output1 = OptimizeFeatureAndEdge.augment(model, feat, strategy='dropedge', p=0.5, edge_index=edge_index, edge_weight=edge_weight)
        output2 = OptimizeFeatureAndEdge.augment(model, feat, strategy='dropedge', p=0.0, edge_index=edge_index, edge_weight=edge_weight)
        output3 = OptimizeFeatureAndEdge.augment(model, feat, strategy='shuffle', edge_index=edge_index, edge_weight=edge_weight)
        loss += (inner(output1, output2) - inner(output2, output3))
        return loss

    @staticmethod
    def augment(model, feat, strategy='dropedge', p=0.5, edge_index=None, edge_weight=None):
        output = None
        if strategy == 'shuffle':
            idx = np.random.permutation(feat.shape[0])
            shuf_fts = feat[idx, :]
            output = model.get_embedding(shuf_fts, edge_index, edge_weight)
        if strategy == "dropedge":
            edge_index, edge_mask = dropout_edge(edge_index, p=p)
            edge_weight = edge_weight[edge_mask]
            output = model.get_embedding(feat, edge_index, edge_weight)
        return output

    @staticmethod
    def sample_random_block(n, eps, edge_index):
        edge_index = edge_index.clone()
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]

        row, col = edge_index[0], edge_index[1]
        edge_index_id = (2 * n - row - 1) * row // 2 + col - row - 1  # // is important to get the correct result
        edge_index_id = edge_index_id.long()
        current_search_space = edge_index_id

        modified_edge_index = linear_to_triu_idx(n, current_search_space)
        perturbed_edge_weight = torch.full_like(current_search_space, eps, dtype=torch.float32, requires_grad=True)
        return modified_edge_index, perturbed_edge_weight


class Backbone(nn.Module):

    def __init__(self, in_feats, h_feats, num_classes):
        super(Backbone, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, num_classes)

    def get_embedding(self, in_feat, edge_index, edge_weight):
        return self.conv1(in_feat, edge_index, edge_weight)

    def forward(self, in_feat, edge_index, edge_weight):
        h = self.conv1(in_feat, edge_index, edge_weight)
        h = F.relu(h)
        h = self.conv2(h, edge_index, edge_weight)
        return h


class GTrans(nn.Module):

    def __init__(self, in_feats, h_feats, num_classes, epochs=2, loop_feat=1, loop_adj=1):
        super(GTrans, self).__init__()
        self.in_feats, self.h_feats = in_feats, h_feats
        self.backbone = Backbone(in_feats, h_feats, num_classes)
        self.purify_model = None
        self.epochs, self.loop_feat, self.loop_adj = epochs, loop_feat, loop_adj

    def purify_graph(self, g, in_feat):
        num_nodes = in_feat.shape[0]
        num_features = self.in_feats
        self.purify_model = OptimizeFeatureAndEdge(num_nodes, num_features, self.epochs, self.loop_feat, self.loop_adj, device=in_feat.device)

        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']

        row, col = g.edges()
        edge_index = torch.cat(
            [row.reshape(1, -1), col.reshape(1, -1)],
            dim=0
        )
        num_edges = edge_index.shape[1]
        edge_weight = torch.ones(size=(num_edges,), dtype=torch.float32, device=edge_index.device)

        in_feat, edge_index, edge_weight, train_mask, labels = in_feat.detach(), edge_index.detach(), edge_weight.detach(), \
            train_mask.detach(), labels.detach()
        n_feat, n_edge_index, n_edge_weight = self.purify_model(self.backbone, in_feat, edge_index, edge_weight, train_mask, labels)
        return n_feat, n_edge_index, n_edge_weight

    def forward(self, g, in_feat, no_grad=False):
        in_feat, edge_index, edge_weight = self.purify_graph(g, in_feat)
        if no_grad:
            with torch.no_grad():
                return self.backbone(in_feat, edge_index, edge_weight)
        return self.backbone(in_feat, edge_index, edge_weight)
