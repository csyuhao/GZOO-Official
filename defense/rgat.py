import torch
import torch.nn.functional as F

import dgl.nn.pytorch as dgl_nn
from dgl import DGLError
import dgl.function as dgl_fn
import dgl.nn.functional as dgl_F
from torch import nn


def edge_similarity(edges):
    num_nodes = edges.src['el'].shape[0]
    flatten_el = edges.src['el'].reshape(num_nodes, -1)
    flatten_er = edges.src['el'].reshape(num_nodes, -1)
    return {'e': F.cosine_similarity(flatten_el, flatten_er, dim=1)}


class RGATConv(dgl_nn.GATConv):

    def __init__(self, in_feats, out_feats, num_heads, threshold=0.1, feat_drop=0., attn_drop=0.,
                 negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False, bias=True):
        super(RGATConv, self).__init__(in_feats, out_feats, num_heads,
                                       feat_drop, attn_drop, negative_slope, residual, activation, allow_zero_in_degree, bias)
        self.threshold = threshold

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        r"""
        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, *, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *, D_{in_{src}})` and :math:`(N_{out}, *, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            A 1D tensor of edge weight values.  Shape: :math:`(|E|,)`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, *, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats
                )
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]

            graph.srcdata.update({"ft": feat_src, "el": feat_src})
            graph.dstdata.update({"er": feat_dst})
            # compute edge attention
            graph.apply_edges(edge_similarity)
            edge_data = graph.edata.pop("e")
            e = torch.log(torch.where(edge_data >= self.threshold, edge_data, 1e-6))
            # compute softmax
            graph.edata["a"] = self.attn_drop(dgl_F.edge_softmax(graph, e))
            if edge_weight is not None:
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(
                    1, self._num_heads, 1
                ).transpose(0, 2)

            # message passing
            graph.update_all(dgl_fn.u_mul_e("ft", "a", "m"), dgl_fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats
                )
                rst = rst + resval
            # bias
            if self.has_explicit_bias:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)),
                    self._num_heads,
                    self._out_feats
                )
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst


class RGAT(torch.nn.Module):
    """ Robust Graph Attention Network from
    Chen et al. UNDERSTANDING AND IMPROVING GRAPH INJECTION ATTACK BY PROMOTING UNNOTICEABILITY. ICLR 2022
    """
    def __init__(self, in_feats, h_feats, out_feats, num_layers, dropout,
                 layer_norm_first=False, threshold=0.1, n_heads=1, att_dropout=0.6):
        super(RGAT, self).__init__()

        self.dropout = dropout
        self.layer_norm_first = layer_norm_first
        self.convs = nn.ModuleList()
        self.convs.append(
            RGATConv(in_feats, h_feats // n_heads, num_heads=n_heads, threshold=threshold, attn_drop=att_dropout)
        )
        self.lns = nn.ModuleList()
        self.lns.append(
            nn.LayerNorm(in_feats)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                RGATConv(h_feats, h_feats // n_heads, num_heads=n_heads, threshold=threshold, attn_drop=att_dropout)
            )
            self.lns.append(
                nn.LayerNorm(h_feats)
            )
        self.lns.append(nn.LayerNorm(h_feats))
        self.convs.append(RGATConv(h_feats, out_feats, num_heads=1, threshold=threshold, attn_drop=att_dropout))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()

    @torch.no_grad()
    def test_forward(self, g, in_feat, edge_weight=None, get_attention=False):
        num_nodes = g.num_nodes()
        x = in_feat
        if self.layer_norm_first:
            x = self.lns[0](in_feat)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(g, x, edge_weight, get_attention)
            x = torch.reshape(x, shape=(num_nodes, -1))
            x = self.lns[i + 1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](g, x)
        x = torch.reshape(x, shape=(num_nodes, -1))
        return x

    def forward(self, g, in_feat, edge_weight=None, get_attention=False, no_grad=False):
        if no_grad:
            return self.test_forward(g, in_feat, edge_weight, get_attention)

        num_nodes = g.num_nodes()
        x = in_feat
        if self.layer_norm_first:
            x = self.lns[0](in_feat)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(g, x, edge_weight, get_attention)
            x = torch.reshape(x, shape=(num_nodes, -1))
            x = self.lns[i + 1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](g, x)
        x = torch.reshape(x, shape=(num_nodes, -1))
        return x
