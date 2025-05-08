import copy

import dgl
import numpy as np
import torch

from dataset.utils import load_dataset, cross_validation_gen

import torch.nn.functional as F


class Dataset(object):

    def __init__(self, dataset_name, *args, **kwargs):

        self.dataset_name = dataset_name
        self.graph, self.discrete_feat = load_dataset(self.dataset_name)
        # normalize graph feature
        self.normalized = kwargs.get('normalized')
        self.origin_discrete_feat = None

        if self.discrete_feat:
            self.origin_discrete_feat = self.graph.ndata['feat'].clone()
        if self.normalized:
            self.graph.ndata['feat'] = F.normalize(self.graph.ndata['feat'])
        self.device = kwargs.get('device')

        if 'train_mask' not in self.graph.ndata:
            train_mask, val_mask, test_mask = cross_validation_gen(self.graph.ndata['label'])
            self.graph.ndata['train_mask'], self.graph.ndata['val_mask'], self.graph.ndata['test_mask'] = \
                train_mask[:, 0], val_mask[:, 0], test_mask[:, 0]
        elif dataset_name == 'wiki_cs':
            self.graph.ndata['train_mask'], self.graph.ndata['val_mask'], self.graph.ndata['test_mask'] = \
                self.graph.ndata['train_mask'][:, 0].bool(), self.graph.ndata['val_mask'][:, 0].bool(), self.graph.ndata['test_mask'].bool()

        self.n_class = int(self.graph.ndata['label'].max().item() + 1)
        self.feature_dim = self.graph.ndata['feat'].shape[1]
        self.degree = self.graph.in_degrees().float().mean().ceil().item()
        self.graph = dgl.add_self_loop(self.graph).to(self.device)

        self.total_test_nodes = self.graph.ndata['test_mask'].sum().item()
        self.feature_budget = (self.graph.ndata['feat'] > 0).float().sum(1).mean() if self.discrete_feat else \
            self.graph.ndata['feat'].sum(1).mean()
        self.mu = self.graph.ndata['feat'].mean(dim=0)
        self.sigma = self.graph.ndata['feat'].std(dim=0)

        self.target_nodes = None
        self.feat_category = None
        self.feat_category_mask = None
        if self.discrete_feat:
            self.init_discrete_category()

    def init_target_nodes(self, model):
        """ target nodes must be correct classified
        """
        g = copy.deepcopy(self.graph)
        test_mask, labels = g.ndata['test_mask'], g.ndata['label']
        logits = model(g, g.ndata['feat'], no_grad=True)
        pred = logits.argmax(dim=1)
        self.target_nodes = test_mask.nonzero(as_tuple=True)[0][(pred[test_mask] == labels[test_mask]).nonzero(as_tuple=True)[0]]

    def init_discrete_category(self):
        """ The category information of discrete data
        """
        feat_arr = self.origin_discrete_feat.cpu().numpy()
        feat_dim = feat_arr.shape[1]

        feat_idx2stats = {}
        maximal_len = 0
        for idx in range(feat_dim):
            feat_idx2stats[idx] = np.unique(feat_arr[:, idx])
            if maximal_len < len(feat_idx2stats[idx]):
                maximal_len = len(feat_idx2stats[idx])

        feat_category = np.zeros(shape=(feat_dim, maximal_len), dtype=np.float32)
        feat_category_mask = np.zeros(shape=(feat_dim, maximal_len), dtype=bool)
        for idx in range(feat_dim):
            cur_feat_dim = len(feat_idx2stats[idx])
            feat_category[idx, :cur_feat_dim] = feat_idx2stats[idx]
            feat_category_mask[idx, :cur_feat_dim] = np.ones(shape=(cur_feat_dim,), dtype=bool)
        self.feat_category = torch.tensor(feat_category, dtype=torch.float32, device=self.device)
        self.feat_category_mask = torch.tensor(feat_category_mask, dtype=torch.bool, device=self.device)
