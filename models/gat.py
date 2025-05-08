import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
import dgl.nn.pytorch as dgl_nn


class GAT(torch.nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads=4, num_layers=2):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        if num_layers > 1:
            for i in range(num_layers):
                if i == 0:
                    self.conv_layers.append(dgl_nn.GATConv(in_size, hid_size, heads, feat_drop=0., attn_drop=0., activation=None))
                elif i == num_layers - 1:
                    self.conv_layers.append(dgl_nn.GATConv(hid_size, out_size, 1, feat_drop=0., attn_drop=0., activation=None))
                else:
                    self.conv_layers.append(dgl_nn.GATConv(hid_size, hid_size, heads, feat_drop=0., attn_drop=0., activation=None))
        else:
             self.conv_layers.append(dgl_nn.GATConv(in_size, out_size, heads, feat_drop=0., attn_drop=0., activation=None))
        self.num_layers = num_layers

    @torch.no_grad()
    def test_forward(self, g, inputs):
        h = inputs
        for i in range(self.num_layers):
            h = self.conv_layers[i](g, h)
            if i == self.num_layers - 1:
                h = torch.flatten(h, 1)
                break
            h = torch.mean(h, dim=1)
            h = F.relu(h)
        return h

    def forward(self, g, inputs, no_grad=False):
        if no_grad:
            return self.test_forward(g, inputs)

        h = inputs
        for i in range(self.num_layers):
            h = self.conv_layers[i](g, h)
            if i == self.num_layers - 1:
                h = torch.flatten(h, 1)
                break
            h = torch.mean(h, dim=1)
            h = F.relu(h)
        return h
