import copy
import logging
from functools import partial

import dgl
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl.nn.pytorch as dgl_nn
from torch import optim
from torch.distributions import Categorical

from attack.base import Attack
from attack.utils import EarlyStop

try:
    if 'logger' not in globals():
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
except NameError:
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def construct_graph(graph, node_feats, node_edges):
    cur_graph = graph.clone()
    cur_num_nodes = graph.num_nodes()

    n_nodes, n_targets = node_edges.shape[0], node_edges.shape[1]
    nodes_vec = torch.arange(0, n_nodes, device=node_feats.device).reshape(-1, 1) + cur_num_nodes

    # add self-loop
    nodes_vec = torch.tile(nodes_vec, dims=(1, n_targets)).reshape(-1)
    node_edges = node_edges.reshape(-1)
    src, dst = torch.cat([nodes_vec, node_edges], dim=-1), torch.cat([node_edges, nodes_vec], dim=-1)
    cur_graph = dgl.add_nodes(cur_graph, n_nodes, data={'feat': node_feats})
    cur_graph = dgl.add_edges(cur_graph, src, dst)
    cur_graph = dgl.add_self_loop(cur_graph)
    return cur_graph


def sample_gumbel(shape, eps=1e-20, device='cuda:0'):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, tau):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector

    https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py
    """
    latent_dim, categorical_dim = logits.shape[0], logits.shape[1]
    y = gumbel_softmax_sample(logits, tau)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)


# ------------------------------------------- [Optimizing For A Network] --------------------------------------------------


def mapping_sub2full(cur_edges, node_mappings):
    index = torch.bucketize(cur_edges.ravel(), node_mappings[0])
    n_edges = node_mappings[1][index].reshape(cur_edges.shape)
    return n_edges


def masked_logits(vec, mask, epsilon=1e-8):
    _vec = vec * mask.float() - 1e4 * (1 - mask.float())
    vec_max = torch.max(_vec, dim=-1, keepdim=True)[0]
    vec = (vec - vec_max) * mask.float()
    return vec


def masked_softmax(vec, mask, dim=-1, epsilon=1e-8):
    # calculate the maximal value
    _vec = vec * mask.float() - 1e4 * (1 - mask.float())
    vec_max = torch.max(_vec, dim=-1, keepdim=True)[0]
    vec = (vec - vec_max) * mask.float()

    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return masked_exps / masked_sums


class CategoricalSampler(object):
    tau = 1e-2

    def __init__(self, tau=None):
        self.tau = tau

    def sample(self, probs):
        sample_ndim, num_classes = probs.ndim, probs.shape[-1]

        sampler = Categorical(probs=probs)
        n_sample_hot = F.one_hot(sampler.sample(), num_classes=num_classes) + self.tau * probs - self.tau * probs.detach()

        # n_sample, n_attrs = probs.shape[0], probs.shape[1]
        # reshaped_probs = probs.reshape(-1, num_classes)
        # sampled_attrs = gumbel_softmax(reshaped_probs, self.tau)
        # if sample_ndim == 2:
        #     n_sample_hot = sampled_attrs.reshape(n_sample, num_classes)
        # else:
        #     n_sample_hot = sampled_attrs.reshape(n_sample, n_attrs, num_classes)

        if sample_ndim == 2:
            indices_matrix = torch.arange(0, num_classes, dtype=torch.long, device=probs.device).reshape(1, -1)
            indices_matrix = torch.tile(indices_matrix, dims=(probs.shape[0], 1))
        elif sample_ndim == 3:
            indices_matrix = torch.arange(0, num_classes, dtype=torch.long, device=probs.device).reshape(1, 1, -1)
            indices_matrix = torch.tile(indices_matrix, dims=(probs.shape[0], probs.shape[1], 1))
        else:
            raise NotImplemented('The number of dimension is {}'.format(sample_ndim))

        n_sample = torch.sum(n_sample_hot * indices_matrix, dim=-1)
        return n_sample, n_sample_hot


class Generator(nn.Module):

    def __init__(self, input_dim, hid_dim, feat_dim, feat_mid_dim, n_nodes, n_edges,
                 tau, discrete_feat, discrete_category, discrete_mask, normalized):
        super(Generator, self).__init__()

        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.discrete_feat = discrete_feat
        self.feat_dim = feat_dim
        self.normalized = normalized
        self.input_dim = input_dim
        self.feat_mid_dim = feat_mid_dim

        if discrete_feat:
            self.discrete_category = nn.Parameter(discrete_category.clone(), requires_grad=False)
            self.discrete_mask = nn.Parameter(
                torch.tile(discrete_mask.clone().unsqueeze(0), dims=(self.n_nodes, 1, 1)),
                requires_grad=False
            )

        self.attr_conv_layers = nn.Sequential(*[
            dgl_nn.GATConv(input_dim, hid_dim // 4, num_heads=4, bias=True, residual=True, activation=None),
            nn.LeakyReLU(),
            dgl_nn.GATConv(2 * hid_dim, n_nodes * feat_dim // 4, num_heads=4, bias=True, residual=True, activation=None),
        ])
        if not self.discrete_feat:
            self.attr_output_layer = nn.Sequential(*[
                nn.LeakyReLU(),
                nn.Linear(in_features=feat_dim, out_features=input_dim),
            ])
        elif self.discrete_feat:
            self.attr_output_layer = nn.Sequential(*[
                nn.LeakyReLU(),
                nn.Linear(in_features=feat_dim, out_features=self.feat_mid_dim * input_dim),
            ])
            self.discrete_prob = nn.Sequential(*[
                nn.LeakyReLU(),
                nn.Linear(in_features=self.feat_mid_dim, out_features=self.discrete_category.shape[1])
            ])

        self.edge_conv_layers = nn.Sequential(*[
            dgl_nn.GATConv(input_dim, hid_dim // 4, num_heads=4, bias=True, residual=True, activation=None),
            nn.LeakyReLU(),
            dgl_nn.GATConv(2 * hid_dim, self.n_nodes, num_heads=1, bias=True, residual=True, activation=None),
        ])

        self.target_feat_layers = nn.Sequential(*[
            nn.Linear(input_dim, 2 * hid_dim),
            nn.LeakyReLU(),
        ])
        self.sampler = CategoricalSampler(tau=tau)

    def sample_edges(self, sample_weight, target_idx):
        num_nodes, num_targets = sample_weight.shape[0], sample_weight.shape[1]
        sample_mask = F.one_hot(target_idx, num_classes=num_targets).bool()

        if self.n_edges == 1:
            sample_edges = torch.tile(target_idx.reshape(-1, 1), dims=(num_nodes, 1)).long()
            return sample_edges

        sample_weights = sample_weight.clone()
        sample_edges = torch.tile(target_idx.reshape(-1, 1), dims=(num_nodes, 1))
        for idx in range(self.n_edges - 1):
            if num_targets <= idx + 1:
                break

            sample_probs = masked_softmax(sample_weights, mask=torch.logical_not(sample_mask), dim=-1)
            new_sample_edges, new_sample_mask = self.sampler.sample(probs=sample_probs)

            # sample_weights = masked_logits(sample_weights, mask=torch.logical_not(sample_mask))
            # new_sample_edges, new_sample_mask = self.sampler.sample(probs=sample_weights)

            new_sample_mask = new_sample_mask.bool()
            sample_mask = torch.logical_or(new_sample_mask, sample_mask)
            sample_edges = torch.cat([sample_edges, new_sample_edges.reshape(num_nodes, 1)], dim=-1)

        return sample_edges.long()

    def sample_attrs(self, sample_weight):
        sample_probs = self.discrete_mask.float() * masked_softmax(sample_weight, mask=self.discrete_mask, dim=-1)
        _, sample_attrs_hot = self.sampler.sample(probs=sample_probs)

        # sample_weights = self.discrete_mask.float() * masked_logits(sample_weight, mask=self.discrete_mask)
        # _, sample_attrs_hot = self.sampler.sample(probs=sample_weights)

        sample_attrs = torch.sum(sample_attrs_hot * self.discrete_category.unsqueeze(0), dim=-1, keepdim=False)
        return sample_attrs

    def forward(self, graph, feat, base_graph, mapping_fn, target_idx):

        layer_num = 0
        attr_feat, edge_feat, target_feat = feat.clone(), feat.clone(), feat[target_idx].clone()
        target_feat = self.target_feat_layers(target_feat)

        for attr_layer, edge_layer in zip(self.attr_conv_layers, self.edge_conv_layers):
            if isinstance(attr_layer, dgl_nn.GraphConv) and isinstance(edge_layer, dgl_nn.GraphConv):
                layer_num += 1
                attr_feat, edge_feat = attr_layer(graph, attr_feat), edge_layer(graph, edge_feat)
                if layer_num == 1:
                    attr_feat, edge_feat = torch.cat([attr_feat, edge_feat], dim=1) + target_feat, torch.cat([edge_feat, attr_feat],
                                                                                                             dim=1) + target_feat
            elif isinstance(attr_layer, dgl_nn.GATConv) and isinstance(edge_layer, dgl_nn.GATConv):
                layer_num += 1
                attr_feat, edge_feat = attr_layer(graph, attr_feat), edge_layer(graph, edge_feat)
                num_attr_node, num_edge_node = attr_feat.shape[0], edge_feat.shape[0]
                attr_feat, edge_feat = attr_feat.reshape(num_attr_node, -1), edge_feat.reshape(num_edge_node, -1)
                if layer_num == 1:
                    attr_feat, edge_feat = torch.cat([attr_feat, edge_feat], dim=1) + target_feat, torch.cat([edge_feat, attr_feat],
                                                                                                             dim=1) + target_feat
            elif isinstance(attr_layer, nn.ReLU) and isinstance(edge_layer, nn.ReLU):
                attr_feat, edge_feat = attr_layer(attr_feat), edge_layer(edge_feat)
            elif isinstance(attr_layer, nn.Tanh) and isinstance(edge_layer, nn.Tanh):
                attr_feat, edge_feat = attr_layer(attr_feat), edge_layer(edge_feat)
            elif isinstance(attr_layer, nn.LeakyReLU) and isinstance(edge_layer, nn.LeakyReLU):
                attr_feat, edge_feat = attr_layer(attr_feat), edge_layer(edge_feat)
            else:
                raise NotImplemented('Not Implemented')

        attr_feat = attr_feat.reshape(-1, self.n_nodes, self.feat_dim)
        attr_feat = torch.mean(attr_feat, dim=0, keepdim=False)
        attr_sample_weights = None
        if self.discrete_feat:
            attr_feat = self.attr_output_layer(attr_feat).reshape(self.n_nodes, self.input_dim, -1)
            attr_sample_weights = self.discrete_prob(attr_feat)
            node_attr = self.sample_attrs(attr_sample_weights)
        else:
            node_attr = self.attr_output_layer(attr_feat)
            attr_sample_weights = node_attr.clone()

        if self.normalized and self.discrete_feat:
            node_attr = F.normalize(node_attr, dim=1)

        edge_sample_weights = edge_feat.t()
        normalized_edge_sample_weights = F.softmax(edge_sample_weights, dim=-1)
        node_edges = self.sample_edges(normalized_edge_sample_weights, target_idx)

        if mapping_fn is not None:
            node_edges = mapping_fn(node_edges)
            cur_graph = construct_graph(base_graph, node_attr, node_edges)
            return node_attr, attr_sample_weights, node_edges, normalized_edge_sample_weights, cur_graph

        cur_graph = construct_graph(graph, node_attr, node_edges)
        return node_attr, attr_sample_weights, node_edges, normalized_edge_sample_weights, cur_graph


def estimate_gradient(target_model, generator, base_graph, cur_graph, node_feats, node_edges, edge_weights, total_sample_num, sigma,
                      mapping_fn, target_idx, node_idx, kappa, target_label, mode='untargeted', discrete_feat=False, normalized=True):
    # Estimate gradients
    device = node_feats.device
    num_nodes, num_targets = edge_weights.shape
    sample_num = total_sample_num // 2
    rand_edge_weights = torch.randn(size=(sample_num, num_nodes, num_targets), dtype=torch.float32, device=device)
    num_nodes, feat_dim = node_feats.shape
    rand_node_attrs = torch.randn(size=(sample_num, num_nodes, feat_dim), dtype=torch.float32, device=device)

    n_node_attrs1 = node_feats.detach().clone().unsqueeze(0) + sigma * rand_node_attrs
    n_node_attrs2 = node_feats.detach().clone().unsqueeze(0) - sigma * rand_node_attrs
    n_node_attrs = torch.cat([n_node_attrs1, n_node_attrs2], dim=0)
    n_edge_weights1 = edge_weights.detach().clone().unsqueeze(0) + sigma * rand_edge_weights
    n_edge_weights2 = edge_weights.detach().clone().unsqueeze(0) - sigma * rand_edge_weights
    n_edge_weights = torch.cat([n_edge_weights1, n_edge_weights2], dim=0)

    # sim = 0
    success = 0

    # ------- Estimate Attribute Generator -----
    loss_list = []
    for idx in range(total_sample_num):
        _cur_graph = construct_graph(base_graph.clone(), n_node_attrs[idx], node_edges)
        _loss, _success = infer_cw_loss(target_model, _cur_graph, target_idx, kappa, target_label, mode)
        if success != 1:
            success = _success
            # if success == 1:
            #     sim = calculate_node_centric_homophony(_cur_graph, target_idx).detach().cpu().numpy()
        loss_list.append(_loss)

    std_loss = np.std(loss_list)
    logger.info('Attribute Gradient Estimation, Sigma = {:.8f}@Loss Std = {:.8f}'.format(sigma, std_loss))

    factor_list = []
    for idx in range(0, sample_num):
        factor_list.append((loss_list[idx] - loss_list[sample_num + idx]) / sigma)
    node_factor = torch.tensor(factor_list, dtype=torch.float32, device=device).reshape(sample_num, 1, 1)
    feat_gradient = torch.mean(node_factor * rand_node_attrs, dim=0, keepdim=False) / 2.

    # ------- Estimate Edge Generator --------
    loss_list = []
    for idx in range(total_sample_num):
        sample_edges = generator.sample_edges(n_edge_weights[idx], node_idx)

        if mapping_fn is not None:
            sample_edges = mapping_fn(sample_edges)
        _cur_graph = construct_graph(base_graph.clone(), node_feats, sample_edges)

        _loss, _success = infer_cw_loss(target_model, _cur_graph, target_idx, kappa, target_label, mode)
        if success != 1:
            success = _success

            # if success == 1:
            #     sim = calculate_node_centric_homophony(_cur_graph, target_idx).detach().cpu().numpy()

        loss_list.append(_loss)

    std_loss = np.std(loss_list)
    logger.info('Edge Gradient Estimation, Sigma = {:.8f}@Loss Std = {:.8f}'.format(sigma, std_loss))

    factor_list = []
    for idx in range(0, sample_num):
        factor_list.append((loss_list[idx] - loss_list[sample_num + idx]) / sigma)

    edge_factor = torch.tensor(factor_list, dtype=torch.float32, device=device).reshape(sample_num, 1, 1)
    edge_gradient = torch.mean(edge_factor * rand_edge_weights, dim=0, keepdim=False) / 2.

    # return success, sim, feat_gradient, edge_gradient
    return success, feat_gradient, edge_gradient


def infer_cw_loss(model, g, target_idx, kappa, target_label, mode='targeted'):
    logits = model(g, g.ndata['feat'], no_grad=True)
    logit = logits[target_idx]
    copy_logit = logit.clone()

    if mode == 'untargeted':
        label = g.ndata['label'][target_idx]
        copy_logit[label] = torch.min(logit) - 10.
        target_logit = logit[label]
        other_logit = torch.max(copy_logit)
        return torch.max(target_logit - other_logit, kappa).item(), logit.argmax().item() != label.item()

    target_logit = logit[target_label]
    copy_logit[target_label] = torch.min(logit) - 10.
    other_logit = torch.max(copy_logit)
    return torch.max(other_logit - target_logit, kappa).item(), logit.argmax().item() == target_label.item()


def cw_loss(model, g, target_idx, kappa, target_label, mode='targeted'):
    logits = model(g, g.ndata['feat'], no_grad=True)
    logit = logits[target_idx]
    copy_logit = logit.clone()

    if mode == 'untargeted':
        label = g.ndata['label'][target_idx]
        copy_logit[label] = torch.min(logit) - 10.
        target_logit = logit[label]
        other_logit = torch.max(copy_logit)
        return torch.max(target_logit - other_logit, kappa), logit.argmax().item() != label.item()

    target_logit = logit[target_label]
    copy_logit[target_label] = torch.min(logit) - 10.
    other_logit = torch.max(copy_logit)
    return torch.max(other_logit - target_logit, kappa), logit.argmax().item() == target_label.item()


def sampling_subgraph(graph, target_idx, khop_edge, device):
    sub_graph, node_idx = dgl.khop_in_subgraph(graph, target_idx, khop_edge, relabel_nodes=True, store_ids=True)

    num_sub_nodes = sub_graph.num_nodes()
    node_mappings = torch.cat(
        [
            torch.arange(0, num_sub_nodes, device=device).reshape(1, -1),
            sub_graph.ndata[dgl.NID].reshape(1, -1)
        ],
        dim=0
    )
    mapping_fn = partial(mapping_sub2full, node_mappings=node_mappings)
    return sub_graph, node_idx, mapping_fn


def evaluate_performance(model, generator, g, target_indexes, khop_edge, attack_results, kappa, test_labels, device, num_succeed):
    model.eval()
    generator.eval()

    num_nodes = len(target_indexes)
    for target_index_idx in range(num_nodes):
        target_idx = target_indexes[target_index_idx]

        if attack_results[target_index_idx].item() == 1:
            continue

        graph = copy.deepcopy(g)
        # sampling a sub-graph
        sub_graph, node_mappings, mapping_fn, node_idx = graph.clone(), None, None, target_idx
        if khop_edge != 0:
            sub_graph, node_idx, mapping_fn = sampling_subgraph(sub_graph, target_idx, khop_edge, device)

        node_attr, attr_sample_weights, node_edges, edge_sample_weights, cur_graph = generator(
            sub_graph, sub_graph.ndata['feat'], graph, mapping_fn, node_idx
        )
        _, success = infer_cw_loss(model, cur_graph, target_idx, kappa, target_label=test_labels, mode='untargeted')

        # if success == 1:
        #     list_sim.append(
        #         (
        #             calculate_node_centric_homophony(cur_graph.clone(), target_idx).detach().cpu().numpy(),
        #             target_idx.detach().cpu().numpy()
        #         )
        #     )

        attack_results[target_index_idx] = success
    success_rate = (torch.sum(attack_results).item() + num_succeed) / (num_succeed + attack_results.shape[0])
    logger.info('Success Rate = {:.2f}%'.format(success_rate * 100.))

    return success_rate, attack_results


# @torch.no_grad()
# def calculate_node_centric_homophony(cur_graph, target_idx=None):
#
#     from torch_geometric.utils.sparse import to_torch_coo_tensor
#     from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
#
#     target_idx = copy.deepcopy(target_idx)
#
#     feat = cur_graph.ndata['feat']
#     cur_graph = dgl.remove_self_loop(cur_graph)
#     row, col = cur_graph.edges()
#     edge_index = torch.cat(
#         [row.reshape(1, -1), col.reshape(1, -1)],
#         dim=0
#     )
#
#     def gcn_norm(adj_t, order=-0.5, add_self_loops=True):
#         if add_self_loops:
#             adj_t = fill_diag(adj_t, 1.0)
#         deg = sparsesum(adj_t, dim=1)
#         deg_inv_sqrt = deg.pow_(order)
#         deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
#         adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
#         adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
#         return adj_t
#
#     edge_attr = torch.tensor([1] * edge_index.shape[1], device=edge_index.device, dtype=torch.float32)
#     edge_index = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_attr)
#     features_propagate = gcn_norm(edge_index, add_self_loops=False) @ feat
#     homophily = F.cosine_similarity(feat, features_propagate)
#     if target_idx is not None:
#         return homophily[target_idx]
#     return homophily


class GZOO(Attack):

    def __init__(self, *args, **kwargs):
        super(GZOO, self).__init__()
        self.early_stop = None
        self.dataset = kwargs.get('dataset')
        self.discrete_feat = self.dataset.discrete_feat
        self.dataset_name = self.dataset.dataset_name
        self.total_test_nodes = self.dataset.total_test_nodes
        self.discrete_category = self.dataset.feat_category
        self.discrete_mask = self.dataset.feat_category_mask
        self.feature_budget_factor = kwargs.get('feature_budget_factor')

        self.khop_edge = kwargs.get('khop_edge')
        self.node_budget = kwargs.get('node_budget')
        self.edge_budget = kwargs.get('edge_budget')

        self.normalized = kwargs.get('normalized')
        self.kappa = kwargs.get('kappa')
        self.sigma = kwargs.get('sigma')
        self.eval_num = kwargs.get('eval_num')
        self.discount_factor = kwargs.get('discount_factor')

        self.device = kwargs.get('device')
        self.victim_model = kwargs.get('victim_model_name')

        self.run_mode = kwargs.get('run_mode')
        self.patience = kwargs.get('patience')
        self.gen_hid_dim = kwargs.get('gen_hid_dim')
        self.gen_feat_dim = kwargs.get('gen_feat_dim')
        self.gen_lr = kwargs.get('gen_lr')
        self.batch_size = kwargs.get('batch_size')
        self.alpha = kwargs.get('alpha')
        self.epochs = kwargs.get('attack_epochs')
        self.tau = kwargs.get('tau')
        self.gen_feat_mid_dim = kwargs.get('gen_feat_mid_dim')

        # feature dim
        self.mu = torch.mean(self.dataset.graph.ndata['feat'], dim=None, keepdim=False)
        self.std = torch.mean(torch.std(self.dataset.graph.ndata['feat'], dim=-1, keepdim=False), dim=None, keepdim=False)
        self.feat_dim = self.dataset.feature_dim
        self.kappa = torch.tensor(self.kappa, dtype=torch.float32, device=self.device)
        self.feature_budget = torch.tile(self.dataset.feature_budget.reshape(1, 1), dims=(self.node_budget, self.feat_dim))

        self.generator, self.optimizer = None, None

    def run_attack_on_batch_dataset(self, model, g, target_indexes, test_labels=None, attack_results=None, num_succeed=0):
        # Random shuffle
        num_nodes = target_indexes.shape[0]
        target_index_indexes = torch.randperm(num_nodes).to(self.device)
        if num_nodes % self.batch_size > 0:
            num_padding_nodes = (num_nodes // self.batch_size + 1) * self.batch_size - num_nodes
            padding_nodes = torch.ones(size=(num_padding_nodes,), dtype=torch.int64, device=self.device) * -1
            target_index_indexes = torch.cat([target_index_indexes, padding_nodes], dim=0)
        target_index_indexes = target_index_indexes.reshape(-1, self.batch_size)
        num_batch = target_index_indexes.shape[0]

        attack_results = copy.deepcopy(attack_results)

        # Optimizing on batch-dataset
        success_rate = 0.
        for batch_idx in range(num_batch):
            batch_target_index_indexes = target_index_indexes[batch_idx]
            logger.info('Optimizing {}-th Batch'.format(batch_idx))

            sample_idx = 0
            for target_index_idx in batch_target_index_indexes:
                target_idx = target_indexes[target_index_idx]

                if target_idx.item() == -1 or attack_results[target_index_idx.item()].item() == 1:
                    continue

                # ----------------------------------------- Training ----------------------------------------
                graph = copy.deepcopy(g)
                # sampling a sub-graph
                sub_graph, node_mappings, mapping_fn, node_idx = graph.clone(), None, None, target_idx
                if self.khop_edge != 0:
                    sub_graph, node_idx, mapping_fn = sampling_subgraph(sub_graph, target_idx, self.khop_edge, self.device)

                self.generator.train()
                self.optimizer.zero_grad()
                node_attr, attr_sample_weights, node_edges, edge_sample_weights, cur_graph = self.generator(
                    sub_graph, sub_graph.ndata['feat'], graph, mapping_fn, node_idx
                )

                # -------------- White-box Optimizing --------------
                success, loss = 0, None
                if self.run_mode == 'white-box':
                    clf_loss, success = cw_loss(model, cur_graph, target_idx, self.kappa, target_label=test_labels, mode='untargeted')
                    feature_loss = torch.tensor(0., device=self.device)
                    if self.discrete_feat:
                        feature_loss += F.mse_loss(
                            node_attr,
                            torch.floor(self.feature_budget_factor * self.feature_budget) / self.feat_dim
                        ) / self.node_budget
                    else:
                        mu, std = torch.mean(node_attr), torch.std(node_attr)
                        feature_loss += (mu - self.mu) ** 2 + (std - self.std) ** 2
                    loss = clf_loss + self.alpha * feature_loss
                    loss.backward()

                    logger.info('Batch Inner Sample Idx = {}@Target Idx = {}@Success = {}@CLF Loss = {:.8f}@Node Loss = {:.8f}'.format(
                        sample_idx, target_idx.item(), success, clf_loss.item(), feature_loss.item()
                    ))

                # -------------- Black-box Optimizing --------------
                if self.run_mode == 'black-box':
                    loss, success = infer_cw_loss(model, cur_graph, target_idx, self.kappa, target_label=test_labels, mode='untargeted')
                    # `success` may be in gradient estimation
                    success, feat_grad, edge_grad = estimate_gradient(
                        model, self.generator, graph, cur_graph, node_attr, node_edges, edge_sample_weights,
                        self.eval_num, self.sigma, mapping_fn, target_idx, node_idx, self.kappa,
                        target_label=test_labels, mode='untargeted', discrete_feat=self.discrete_feat
                    )

                    feature_loss = torch.tensor(0., device=self.device)
                    if self.discrete_feat:
                        feature_loss += F.mse_loss(
                            node_attr,
                            torch.floor(self.feature_budget_factor * self.feature_budget) / self.feat_dim
                        ) / self.node_budget
                    else:
                        mu, std = torch.mean(node_attr), torch.std(node_attr)
                        feature_loss += (mu - self.mu) ** 2 + (std - self.std) ** 2
                    feature_loss = self.alpha * feature_loss
                    feature_loss.backward(retain_graph=True)

                    # Estimated gradients
                    if self.edge_budget > 1:
                        node_attr.backward(feat_grad, retain_graph=True)
                        edge_sample_weights.backward(edge_grad)
                    else:
                        node_attr.backward(feat_grad)
                    logger.info(
                        'Batch Inner Sample Idx = {}@Target Idx = {}@Success = {}@CLF Loss = {:.8f}@Node Regular Loss = {:.8f}'.format(
                            sample_idx, target_idx.item(), success, loss, feature_loss.item()
                        ))

                self.optimizer.step()
                attack_results[target_index_idx.item()] = success
                sample_idx += 1

            #  ------------------------ Evaluating ------------------------
            success_rate, attack_results = evaluate_performance(model, self.generator, g, target_indexes,
                                                                self.khop_edge, attack_results, self.kappa, test_labels, self.device, num_succeed)
        return success_rate, attack_results

    def run_attack(self, model, g, target_indexes, test_labels=None, wandb_ins=None):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        success_rate = 0.
        cur_target_indexes = copy.deepcopy(target_indexes)

        logger.info('Total Edge = {}'.format(self.edge_budget))
        self.generator = Generator(
            input_dim=self.feat_dim, hid_dim=self.gen_hid_dim, feat_dim=self.gen_feat_dim, feat_mid_dim=self.gen_feat_mid_dim,
            n_nodes=self.node_budget, n_edges=self.edge_budget, tau=self.tau, discrete_feat=self.discrete_feat,
            discrete_category=self.discrete_category, discrete_mask=self.discrete_mask, normalized=self.normalized
        ).to(self.device)
        self.optimizer = optim.Adam(params=self.generator.parameters(), lr=self.gen_lr)

        edge_budget = self.edge_budget
        prev_attack_results = torch.zeros_like(target_indexes).to(self.device)
        his_attack_results = torch.zeros_like(target_indexes).to(self.device)
        for e in range(1, edge_budget + 1):
            logger.info('Cur Edge Budget = {}'.format(e))
            self.edge_budget = e

            self.early_stop = EarlyStop(patience=self.patience, epsilon=1e-4)
            for epoch in range(self.epochs):
                logger.info('Epoch = {}'.format(epoch))
                success_rate, cur_attack_results = self.run_attack_on_batch_dataset(
                    model, g, cur_target_indexes, test_labels=test_labels,
                    attack_results=prev_attack_results, num_succeed=torch.sum(his_attack_results).item()
                )

                cur_target_indexes = cur_target_indexes[cur_attack_results == 0]
                his_attack_results[his_attack_results == 0] = cur_attack_results
                prev_attack_results = cur_attack_results[cur_attack_results == 0]

                if not isinstance(self.early_stop, bool):
                    self.early_stop(100.0 - success_rate * 100.)
                    if self.early_stop.stop:
                        break

            logger.info('Edge Budget = {}@Attack Success Rate = {:.2f}%'.format(e, success_rate * 100.))

        logger.info('Final Success Rate = {:.2f}%'.format(success_rate * 100.))
