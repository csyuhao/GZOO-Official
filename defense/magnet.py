import warnings
import math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GATConv, GCNConv
from torch_geometric.utils import get_laplacian, degree

warnings.filterwarnings('ignore')


def soft_thresholding(x, soft_eta, mode):
    """
    Perform row-wise soft thresholding.
    The row wise shrinkage is specific on E(k+1) updating
    The element wise shrinkage is specific on Z(k+1) updating

    :param x: one block of target matrix, shape[num_nodes, num_features]
    :param soft_eta: threshold scalar stores in a torch tensor
    :param mode: model types selection "row" or "element"
    :return: one block of matrix after shrinkage, shape[num_nodes, num_features]

    """
    assert mode in ('element', 'row'), 'shrinkage type is invalid (element or row)'
    if mode == 'row':
        row_norm = torch.linalg.norm(x, dim=1).unsqueeze(1)
        row_norm.clamp_(1e-12)
        row_thresh = (F.relu(row_norm - soft_eta) + soft_eta) / row_norm
        out = x * row_thresh
    else:
        out = F.relu(x - soft_eta) - F.relu(-x - soft_eta)

    return out


def compute_wtv(w, v):
    wtv = torch.zeros_like(v[0])
    for i in range(len(v)):
        wtv += torch.sparse.mm(w[i], v[i])
    return wtv


def hard_thresholding(x, soft_eta, mode):
    """
    Perform row-wise hard thresholding.
    The row wise shrinkage is specific on E(k+1) updating
    The element wise shrinkage is specific on Z(k+1) updating

    :param x: one block of target matrix, shape[num_nodes, num_features]
    :param soft_eta: threshold scalar stores in a torch tensor
    :param mode: model types selection "row" or "element"
    :return: one block of matrix after shrinkage, shape[num_nodes, num_features]

    """
    assert mode in ('element', 'row'), 'shrinkage type is invalid (element or row)'
    tmp = torch.zeros_like(x)
    tmp[x - soft_eta > 0] = 1
    tmp[-x - soft_eta > 0] = 1
    out = x * tmp
    return out


def ChebyshevApprox(f, n):  # assuming f : [0, pi] -> R
    quad_points = 500
    c = np.zeros(n)
    a = np.pi / 2
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(a * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)

    return c


def get_operator(L, DFilters, n, s, J, Lev, device='cuda'):
    r = len(DFilters)
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)
    a = np.pi / 2  # consider the domain of masks as [0, pi]
    # Fast Tight Frame Decomposition (FTFD)
    FD1 = torch.eye(L.shape[0], dtype=torch.float32, device=device)
    for j in range(r):
        c[j] = torch.from_numpy(c[j]).to(device)
    values = L.data
    indices = np.vstack((L.row, L.col))
    i = torch.LongTensor(indices).to(device)
    v = torch.FloatTensor(values).to(device)
    L = torch.sparse_coo_tensor(i, v, L.shape).to(device)

    d = dict()
    for l in range(1, Lev + 1):
        for j in range(r):
            T0F = FD1
            T1F = ((s ** (-J + l - 1) / a) * L) @ T0F - T0F
            d[j, l - 1] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J + l - 1)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l - 1] += c[j][k] * TkF
        FD1 = d[0, l - 1]
    return d


@torch.no_grad()
def scipy_to_torch_sparse(A):
    A = sparse.coo_matrix(A)
    row = torch.tensor(A.row)
    col = torch.tensor(A.col)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data)

    return torch.sparse_coo_tensor(index, value, A.shape)


class GraphEncoder(nn.Module):

    def __init__(self, feat_dim, hidden_dim, dropout=0.5):
        super(GraphEncoder, self).__init__()

        self.gc1 = GATConv(feat_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gc2 = GATConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.bn2(x)

        return x


class AttributeDecoder(nn.Module):

    def __init__(self, feat_dim, hidden_dim, dropout=0.5):
        super(AttributeDecoder, self).__init__()
        self.gc1 = GATConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gc2 = GATConv(hidden_dim, feat_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        return x


class Dominant(nn.Module):

    def __init__(self, feat_dim, hidden_dim, dropout=0.5):
        super(Dominant, self).__init__()
        self.shared_encoder = GraphEncoder(feat_dim, hidden_dim, dropout)
        self.attr_decoder = AttributeDecoder(feat_dim, hidden_dim, dropout)

    def forward(self, feat, edge_index):
        x = self.shared_encoder(feat, edge_index)
        x = self.attr_decoder(x, edge_index)
        return x


class ReconGraph(object):

    def __init__(self, in_feats, h_feats, device='cuda', epochs=1, lr=0.01):
        self.dominant = Dominant(in_feats, h_feats).to(device)
        self.epochs = epochs
        self.lr = lr

    def __call__(self, feat, edge_index):
        optimizer = torch.optim.Adam(self.dominant.parameters(), lr=self.lr)
        self.dominant.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            X_hat = self.dominant(feat, edge_index)
            loss = F.mse_loss(X_hat, feat)
            loss.backward()
            optimizer.step()

        self.dominant.eval()
        return self.dominant(feat, edge_index)


class NodeDenoisingADMM(nn.Module):

    def __init__(self, num_nodes, num_features, r, J, nu, admm_iter, rho, gamma_0):
        super(NodeDenoisingADMM, self).__init__()
        self.r = r
        self.J = J
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.admm_iter = admm_iter
        self.rho = rho
        self.nu = [nu] * J
        for i in range(J):
            self.nu[i] = self.nu[i] / np.power(4.0, i)
        self.nu = [0.0] + self.nu
        self.gamma_max = 1e+6
        self.initial_gamma = gamma_0
        self.gamma = self.initial_gamma

    def forward(self, F, W_list, d, mask, init_Zk=None, init_Yk=None, lp=1, lq=2, boost=False, stop_thres=0.05, boost_value=4, thres_iter=15):
        """
        Parameters
        ----------
        F : Graph signal to be smoothed, shape [Num_node, Num_features].
        W_list : Framelet Base Operator, in list, each is a sparse matrix of size Num_node x Num_node.
        d : Vector of normalized graph node degrees in shape [Num_node, 1].
        init_Zk: Initialized list of (length: j * l) zero matrix in shape [Num_node, Num_feature].
        init_Yk: Initialized lists of (length: j*l) zero matrix in shape [Num_node, Num_feature].

        :returns:  Smoothed graph signal U

        """
        if init_Zk is None:
            Zk = []
            for j in range(self.r - 1):
                for l in range(self.J):
                    Zk.append(torch.zeros(torch.Size([self.num_nodes, self.num_features])).to(F.device))
            Zk = [torch.zeros((self.num_nodes, self.num_features)).to(F.device)] + Zk
        else:
            Zk = init_Zk
        if init_Yk is None:
            Yk = []
            for j in range(self.r - 1):
                for l in range(self.J):
                    Yk.append(torch.zeros(torch.Size([self.num_nodes, self.num_features])).to(F.device))
            Yk = [torch.zeros((self.num_nodes, self.num_features)).to(F.device)] + Yk
        else:
            Yk = init_Yk

        self.gamma = self.initial_gamma
        vk = [Yk_jl for Zk_jl, Yk_jl in zip(Zk, Yk)]
        Uk = F
        k = 1
        ak = boost_value
        v_til = vk
        energy_list = []
        diff_list = []
        while k < thres_iter:
            if lp == 1:
                Zk = [soft_thresholding(torch.sparse.mm(W_jl, Uk) + Yk_jl / self.gamma, (nu_jl / self.gamma) * d.unsqueeze(1), 'element')
                      for nu_jl, W_jl, Yk_jl in zip(self.nu, W_list, Yk)]
            if lp == 0:
                Zk = [hard_thresholding(torch.sparse.mm(W_jl, Uk) + Yk_jl / self.gamma, (nu_jl / self.gamma) * d.unsqueeze(1), 'element')
                      for nu_jl, W_jl, Yk_jl in zip(self.nu, W_list, Yk)]
            if lq == 2:
                if boost == 0:
                    v_til = [Yk_jl - self.gamma * Zk_jl for Zk_jl, Yk_jl in zip(Zk, Yk)]
                if boost == 1:
                    boosta = (1 + math.sqrt(1 + 4 * ak * ak)) / 2
                    v_til = [Yk_jl - self.gamma * Zk_jl for Zk_jl, Yk_jl in zip(Zk, Yk)]
                    v_til = [item + ((ak - 1) / boosta) * (item - item0) for item0, item in zip(v_til0, v_til)]
                    v_til0 = v_til
                WTV = compute_wtv(W_list, v_til)
                Uk = (d.unsqueeze(1) * F * mask * mask - WTV) / (d.unsqueeze(1) * mask * mask + self.gamma)
            if lq == 1:
                if boost == 0:
                    v_til = [Yk_jl - self.gamma * Zk_jl for Zk_jl, Yk_jl in zip(Zk, Yk)]  ####
                if boost == 1:
                    boosta = (1 + math.sqrt(1 + 4 * ak * ak)) / 2
                    v_til = [Yk_jl - self.gamma * Zk_jl for Zk_jl, Yk_jl in zip(Zk, Yk)]
                    v_til = [item + ((ak - 1) / boosta) * (item - item0) for item0, item in zip(v_til0, v_til)]
                    v_til0 = v_til
                WTV = compute_wtv(W_list, v_til)
                Yk = soft_thresholding(-F - WTV / self.gamma, (1 / 2 * self.gamma) * d.unsqueeze(1), 'element')
                Yk = mask * Yk + (1 - mask) * (-F - WTV / self.gamma)
                Uk = Yk + F
            if boost == 0:
                Yk = [Yk_jl + self.gamma * (torch.sparse.mm(W_jl, Uk) - Zk_jl) for W_jl, Yk_jl, Zk_jl in
                      zip(W_list, Yk, Zk)]
            if boost == 1:
                boosta = (1 + math.sqrt(1 + 4 * ak * ak)) / 2
                Y0 = Yk
                Yk = [Yk_jl + self.gamma * (torch.sparse.mm(W_jl, Uk) - Zk_jl) for W_jl, Yk_jl, Zk_jl in
                      zip(W_list, Yk, Zk)]
                Yk = [item1 + ((ak - 1) / boosta) * (item1 - item0) for item1, item0 in zip(Yk, Y0)]
                ak = boosta

            k += 1
            if k > thres_iter:
                break
        return Uk, energy_list, diff_list


class Backbone(nn.Module):

    def __init__(self, in_feats, h_feats, num_classes):
        super(Backbone, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, num_classes)

    def get_embedding(self, in_feat, edge_index):
        return self.conv1(in_feat, edge_index)

    def forward(self, in_feat, edge_index):
        h = self.conv1(in_feat, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        return h


class MAGNet(nn.Module):

    def __init__(self, in_feats, h_feats, num_classes, device='cuda', anomaly_ratio=0.01, lp=1, lq=1,
                 n=6, s=4, Lev=1, nu=10, admm_iter=1, rho=0.95, mu2_0=10, boost=0, stop_thres=1, boost_value=0.3, thres_iter=1):
        super(MAGNet, self).__init__()
        self.anomaly_ratio = anomaly_ratio
        self.backbone = Backbone(in_feats, h_feats, num_classes).to(device)
        self.purify_model = ReconGraph(in_feats, h_feats, device=device)
        self.n, self.s, self.Lev = n, s, Lev
        self.in_feats, self.lp, self.lq = in_feats, lp, lq
        self.nu, self.admm_iter, self.rho, self.mu2_0 = nu, admm_iter, rho, mu2_0
        self.boost, self.stop_thres, self.boost_value, self.thres_iter = boost, stop_thres, boost_value, thres_iter

    def purify_graph(self, g, in_feat):
        row, col = g.edges()
        edge_index = torch.cat(
            [row.reshape(1, -1), col.reshape(1, -1)],
            dim=0
        )

        # Reconstruction Graph
        in_feat, edge_index = in_feat.detach(), edge_index.detach()
        in_feat_hat = self.purify_model(in_feat, edge_index)
        diff = (in_feat - in_feat_hat).detach().cpu().numpy()
        mask = np.ones_like(in_feat.cpu().numpy())
        mask[diff > self.anomaly_ratio] = 0

        # Building Mask
        num_nodes = in_feat.shape[0]
        L = get_laplacian(edge_index, num_nodes=num_nodes, normalization='sym')
        L = sparse.coo_matrix((L[1].cpu().numpy(), (L[0][0, :].cpu().numpy(), L[0][1, :].cpu().numpy())), shape=(num_nodes, num_nodes))

        # get maximum eigenvalues of the graph Laplacian
        lobpcg_init = np.random.rand(num_nodes, 1)
        lambda_max, _ = lobpcg(L, lobpcg_init, maxiter=5, verbosityLevel=0)
        lambda_max = lambda_max[0]

        # get degrees
        deg = degree(edge_index[0], num_nodes).to(in_feat.device)

        D1 = lambda x: np.cos(x / 2)
        D2 = lambda x: np.sin(x / 2)
        DFilters = [D1, D2]
        RFilters = [D1, D2]

        J = np.log(lambda_max / np.pi) / np.log(self.s) + self.Lev - 1  # dilation level to start the decomposition
        d = get_operator(L, DFilters, self.n, self.s, J, self.Lev, device=in_feat.device)

        r = len(DFilters)
        d_list = list()
        for i in range(r):
            for l in range(self.Lev):
                d_list.append(d[i, l])

        device = in_feat.device
        W_list = [d.to(device) for d in d_list[self.Lev - 1:]]
        # initialize the denoising filter
        smoothing = NodeDenoisingADMM(num_nodes, self.in_feats, r, self.Lev, self.nu, self.admm_iter, self.rho, self.mu2_0).to(device)

        out, energy, diff = smoothing(in_feat, W_list=W_list, d=deg,
                                      mask=torch.from_numpy(mask).to(device), lp=self.lp, lq=self.lq, boost=self.boost,
                                      stop_thres=self.stop_thres, boost_value=self.boost_value, thres_iter=self.thres_iter)
        out_data = out.detach()
        return out_data, edge_index

    def forward(self, g, in_feat, no_grad=False):
        if no_grad:
            row, col = g.edges()
            edge_index = torch.cat(
                [row.reshape(1, -1), col.reshape(1, -1)],
                dim=0
            )
            with torch.no_grad():
                return self.backbone(in_feat, edge_index)
        in_feat, edge_index = self.purify_graph(g, in_feat)
        return self.backbone(in_feat, edge_index)
