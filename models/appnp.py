import dgl
import torch


class APPNP(torch.nn.Module):
    # code copied from dgl examples
    def __init__(self, in_feats, h_feats, num_classes):
        super(APPNP, self).__init__()
        self.mlp = torch.nn.Linear(in_feats, num_classes)
        self.conv = dgl.nn.APPNPConv(k=3, alpha=0.5)

    @torch.no_grad()
    def test_forward(self, g, in_feat):
        in_feat = self.mlp(in_feat)
        h = self.conv(g, in_feat)
        return h

    def forward(self, g, in_feat, no_grad=False):
        if no_grad:
            return self.test_forward(g, in_feat)

        in_feat = self.mlp(in_feat)
        h = self.conv(g, in_feat)
        return h
