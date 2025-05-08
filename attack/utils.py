import dgl
import numpy as np
import scipy
import torch
from torch_sparse import SparseTensor
from scipy.sparse.csr import csr_matrix
import torch_sparse
from sklearn.metrics import roc_auc_score


def eval_acc(pred, labels, mask=None):
    r"""

    Description
    -----------
    Accuracy metric for node classification.

    Parameters
    ----------
    pred : torch.Tensor
        Output logits of model in form of ``N * 1``.
    labels : torch.LongTensor
        Labels in form of ``N * 1``.
    mask : torch.Tensor, optional
        Mask of nodes to evaluate in form of ``N * 1`` torch bool tensor. Default: ``None``.

    Returns
    -------
    acc : float
        Node classification accuracy.

    """

    if mask is not None:
        pred, labels = pred[mask], labels[mask]
        if pred is None or labels is None:
            return 0.0

    acc = (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

    return acc


def eval_rocauc(pred, labels, mask=None):
    r"""

    Description
    -----------
    ROC-AUC score for multi-label node classification.

    Parameters
    ----------
    pred : torch.Tensor
        Output logits of model in form of ``N * 1``.
    labels : torch.LongTensor
        Labels in form of ``N * 1``.
    mask : torch.Tensor, optional
        Mask of nodes to evaluate in form of ``N * 1`` torch bool tensor. Default: ``None``.


    Returns
    -------
    rocauc : float
        Average ROC-AUC score across different labels.

    """

    rocauc_list = []
    if mask is not None:
        pred, labels = pred[mask], labels[mask]
        if pred is None or labels is None:
            return 0.0
    pred = pred.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    for i in range(labels.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(labels[:, i] == 1) > 0 and np.sum(labels[:, i] == 0) > 0:
            rocauc_list.append(roc_auc_score(y_true=labels[:, i], y_score=pred[:, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    rocauc = sum(rocauc_list) / len(rocauc_list)

    return rocauc


def eval_f1multilabel(pred, labels, mask=None):
    r"""

    Description
    -----------
    F1 score for multi-label node classification.

    Parameters
    ----------
    pred : torch.Tensor
        Output logits of model in form of ``N * 1``.
    labels : torch.LongTensor
        Labels in form of ``N * 1``.
    mask : torch.Tensor, optional
        Mask of nodes to evaluate in form of ``N * 1`` torch bool tensor. Default: ``None``.


    Returns
    -------
    f1 : float
        Average F1 score across different labels.

    """

    if mask is not None:
        pred, labels = pred[mask], labels[mask]
        if pred is None or labels is None:
            return 0.0
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    tp = (labels * pred).sum().float()
    fp = ((1 - labels) * pred).sum().float()
    fn = (labels * (1 - pred)).sum().float()

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = (2 * precision * recall) / (precision + recall + epsilon)
    f1 = f1.item()

    return f1


def get_weights_arithmetic(n, w_1, order='a'):
    r"""

    Description
    -----------
    Arithmetic weights for calculating weighted robust score.

    Parameters
    ----------
    n : int
        Number of scores.
    w_1 : float
        Initial weight of the first term.
    order : str, optional
        ``a`` for ascending order, ``d`` for descending order. Default: ``a``.

    Returns
    -------
    weights : list
        List of weights.

    """

    weights = []
    epsilon = 2 / (n - 1) * (1 / n - w_1)
    for i in range(1, n + 1):
        weights.append(w_1 + (i - 1) * epsilon)

    if 'd' in order:
        weights.reverse()

    return weights


def get_weights_polynomial(n, p=2, order='a'):
    r"""

    Description
    -----------
    Arithmetic weights for calculating weighted robust score.

    Parameters
    ----------
    n : int
        Number of scores.
    p : float
        Power of denominator.
    order : str, optional
        ``a`` for ascending order, ``d`` for descending order. Default: ``a``.

    Returns
    -------
    weights_norms : list
        List of normalized polynomial weights.

    """

    weights = []
    for i in range(1, n + 1):
        weights.append(1 / i ** p)
    weights_norm = [weights[i] / sum(weights) for i in range(n)]
    if 'a' in order:
        weights_norm = weights_norm[::-1]

    return weights_norm


class EarlyStop(object):
    r"""

    Description
    -----------
    Strategy to early stop attack process.

    """
    def __init__(self, patience=100, epsilon=1e-4):
        r"""

        Parameters
        ----------
        patience : int, optional
            Number of epoch to wait if no further improvement. Default: ``1000``.
        epsilon : float, optional
            Tolerance range of improvement. Default: ``1e-4``.

        """
        self.patience = patience
        self.epsilon = epsilon
        self.min_score = None
        self.stop = False
        self.count = 0

    def __call__(self, score):
        r"""

        Parameters
        ----------
        score : float
            Value of attack score.

        """
        if self.min_score is None:
            self.min_score = score
        elif self.min_score - score > 0:
            self.count = 0
            self.min_score = score
        elif self.min_score - score < self.epsilon:
            self.count += 1
            if self.count > self.patience:
                self.stop = True

    def reset(self):
        self.min_score = None
        self.stop = False
        self.count = 0


def inject_node(g, feat):
    nid = g.num_nodes()
    g = dgl.add_nodes(g, 1, {'feat': feat.reshape(1, -1)})
    g = dgl.add_edges(g, nid, nid)  # add self loop

    return g


def wire_edge(g, dst):
    g = dgl.add_edges(
        g,
        torch.tensor([dst, g.number_of_nodes() - 1]).to(g.device),
        torch.tensor([g.number_of_nodes() - 1, dst]).to(g.device)
    )
    return g


def inject_multi_nodes(g, feat):
    nid = g.num_nodes()
    num_nodes = feat.shape[0]
    g = dgl.add_nodes(g, num_nodes, {'feat': feat})
    for i in range(num_nodes):
        g = dgl.add_edges(g, nid + i, nid + i)  # add self loop

    return g


def wire_multi_edges(g, src, dst_list):
    for dst in dst_list:
        g = dgl.add_edges(
            g,
            torch.tensor([dst, src]).to(g.device),
            torch.tensor([src, dst]).to(g.device)
        )
    return g


def feat_preprocess(features, device='cpu'):
    r"""

    Description
    -----------
    Preprocess the features.

    Parameters
    ----------
    features : torch.Tensor or numpy.ndarray
        Features in form of torch tensor or numpy array.
    feat_norm : str, optional
        Type of features normalization, choose from ["arctan", "tanh", None]. Default: ``None``.
    device : str, optional
        Device used to host data. Default: ``cpu``.

    Returns
    -------
    features : torch.Tensor
        Features in form of torch tensor on chosen device.

    """

    if type(features) != torch.Tensor:
        features = torch.FloatTensor(features)
    elif features.type() != 'torch.FloatTensor':
        features = features.float()

    features = features.to(device)

    return features


def adj_to_tensor(adj):
    r"""

    Description
    -----------
    Convert adjacency matrix in scipy sparse format to torch sparse tensor.

    Parameters
    ----------
    adj : scipy.sparse.csr.csr_matrix
        Adjacency matrix in form of ``N * N`` sparse matrix.
    Returns
    -------
    adj_tensor : torch.Tensor
        Adjacency matrix in form of ``N * N`` sparse tensor.

    """

    if type(adj) != scipy.sparse.coo.coo_matrix:
        adj = adj.tocoo()

    sparse_row = torch.LongTensor(adj.row)
    sparse_col = torch.LongTensor(adj.col)
    N = adj.shape[0]
    perm = (sparse_col * N + sparse_row).argsort()
    sparse_row, sparse_col = sparse_row[perm], sparse_col[perm]
    adj_tensor = SparseTensor(row=sparse_col, col=sparse_row, value=None, sparse_sizes=torch.Size(adj.shape), is_sorted=True).to_symmetric()
    return adj_tensor


def tensor_to_adj(adj_tensor):
    if type(adj_tensor) is tuple:
        # TODO: BUG
        M = len(adj_tensor[0])
        # DGL graph
        adj = csr_matrix((torch.ones(M), adj_tensor), (M, M), np.int64)
    elif type(adj_tensor) is torch_sparse.tensor.SparseTensor:
        row, col, data = adj_tensor.cpu().coo()
        if data is None:
            data = torch.ones(len(row))
        M = adj_tensor.size(0)
        adj = csr_matrix((torch.ones(len(row)).numpy(), (row.numpy(), col.numpy())), (M, M), np.float64)
    else:
        # TODO: BUG
        M = len(adj_tensor[0])
        # PyG graph
        adj = csr_matrix(torch.ones(M), (adj_tensor[0], adj_tensor[1]), (M, M), np.int64)

    return adj
