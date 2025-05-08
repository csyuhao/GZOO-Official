import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GCNConv


class ReconAdj(nn.Module):

    def __init__(self, epochs=1, lr=0.01, gamma=1.0):
        super(ReconAdj, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma

    def recon_edge_weight(self, model, feat, edge_index, train_mask, labels):
        model.eval()

        num_edges = edge_index.shape[1]
        estimated_edge_weight = nn.Parameter(torch.ones((num_edges,), dtype=torch.float32, device=feat.device))
        estimated_edge_weight.requires_grad = True
        optimizer = torch.optim.Adam([estimated_edge_weight], lr=self.lr)
        for idx in range(self.epochs):
            optimizer.zero_grad()
            logits = model(feat, edge_index, edge_weight=estimated_edge_weight)

            loss_gcn = F.cross_entropy(logits[train_mask], labels[train_mask])
            loss_fro = torch.norm(estimated_edge_weight, p='fro')
            loss = - loss_fro + self.gamma * loss_gcn

            loss.backward()
            optimizer.step()
            estimated_edge_weight.data.copy_(torch.clamp(estimated_edge_weight, 0, 1))
        return estimated_edge_weight.detach()


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


class ProGNN(nn.Module):
    """ ProGNN (Properties Graph Neural Network).
        See more details in Graph Structure Learning for Robust Graph Neural Networks, KDD 2020, https://arxiv.org/abs/2005.10203.
    """

    def __init__(self, in_feats, h_feats, num_classes, device='cuda'):
        super(ProGNN, self).__init__()
        self.backbone = Backbone(in_feats, h_feats, num_classes).to(device)
        self.recon_model = ReconAdj()

    def forward(self, g, in_feat, no_grad=False):
        # Get the estimated edge weight
        row, col = g.edges()
        edge_index = torch.cat(
            [row.reshape(1, -1), col.reshape(1, -1)],
            dim=0
        )
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        in_feat = in_feat.detach()
        estimated_edge_weight = self.recon_model.recon_edge_weight(self.backbone, in_feat, edge_index, train_mask, labels)
        if no_grad:
            with torch.no_grad():
                return self.backbone(in_feat, edge_index, estimated_edge_weight)
        return self.backbone(in_feat, edge_index, estimated_edge_weight)
