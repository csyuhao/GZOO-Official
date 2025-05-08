import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dgl_nn


class SGC(torch.nn.Module):

    # code copied from dgl examples
    def __init__(self, in_feats, h_feats, num_classes, num_layers):
        super(SGC, self).__init__()

        self.conv_layers = nn.ModuleList()
        if num_layers > 1:
            for i in range(num_layers):
                if i == 0:
                    self.conv_layers.append(dgl_nn.SGConv(in_feats, h_feats, k=2))
                elif i == num_layers - 1:
                    self.conv_layers.append(dgl_nn.SGConv(h_feats, num_classes, k=2))
                else:
                    self.conv_layers.append(dgl_nn.SGConv(h_feats, h_feats, k=2))
        else:
            self.conv_layers.append(dgl_nn.SGConv(in_feats, num_classes, k=2))
        self.num_layers = num_layers

    @torch.no_grad()
    def test_forward(self, g, in_feat):
        h = in_feat
        for i in range(self.num_layers):
            h = self.conv_layers[i](g, h)
            if i == self.num_layers - 1:
                break
            h = F.relu(h)
        return h

    def forward(self, g, in_feat, no_grad=False):
        if no_grad:
            return self.test_forward(g, in_feat)

        h = in_feat
        for i in range(self.num_layers):
            h = self.conv_layers[i](g, h)
            if i == self.num_layers - 1:
                break
            h = F.relu(h)
        return h
