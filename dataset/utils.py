import dgl
import numpy as np
import scipy.sparse as sp
import torch


def load_npz(dataset):
    if dataset == 'reddit':
        dataset = '12k_reddit'
    elif dataset == 'ogbproducts':
        dataset = '10k_ogbproducts'
    else:
        raise Exception('Not implemented err.')

    file_name = 'data/{}.npz'.format(dataset)
    if not file_name.endswith('.npz'):
        file_name += '.npz'

    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix(
            (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
            shape=loader['adj_shape']
        )

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix(
                (loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                shape=loader['attr_shape']
            )
        else:
            attr_matrix = None
        labels = loader.get('labels')

    g = dgl.graph(adj_matrix.nonzero())
    g = dgl.add_self_loop(g)
    g.ndata['feat'] = torch.tensor(attr_matrix.todense()).float()
    split = np.load('data/{}_split.npy'.format(dataset), allow_pickle=True).item()
    train_mask, val_mask, test_mask = split['train'], split['val'], split['test']
    temp = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
    temp[train_mask] = True
    g.ndata['train_mask'] = temp
    temp = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
    temp[val_mask] = True
    g.ndata['val_mask'] = temp
    temp = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
    temp[test_mask] = True
    g.ndata['test_mask'] = temp

    g.ndata['label'] = torch.tensor(labels)
    return g


def load_dataset(dataset_name):
    """ Loading dataset
    """

    if dataset_name == 'reddit':
        g = load_npz('reddit')                  # dgl.data.RedditDataset()[0]
        discrete_feat = False
    elif dataset_name == 'ogbproducts':
        g = load_npz('ogbproducts')
        discrete_feat = False
    elif dataset_name == 'cora':
        g = dgl.data.CoraGraphDataset(verbose=False)[0]
        discrete_feat = True
    elif dataset_name == 'citeseer':
        g = dgl.data.CiteseerGraphDataset(verbose=False)[0]
        discrete_feat = True
    elif dataset_name == 'pubmed':
        g = dgl.data.PubmedGraphDataset(verbose=False)[0]
        discrete_feat = False
    elif dataset_name == 'wiki_cs':
        g = dgl.data.WikiCSDataset(verbose=False)[0]
        discrete_feat = False
    elif dataset_name == 'co_computer':
        g = dgl.data.AmazonCoBuyComputerDataset(verbose=False)[0]
        discrete_feat = True
    elif dataset_name == 'co_photo':
        g = dgl.data.AmazonCoBuyPhotoDataset(verbose=False)[0]
        discrete_feat = True
    elif dataset_name == 'CoauthorCS':
        g = dgl.data.CoauthorCSDataset(verbose=False)[0]
        discrete_feat = True
    elif dataset_name == 'CoauthorPhysics':
        g = dgl.data.CoauthorPhysicsDataset(verbose=False)[0]
        discrete_feat = True
    elif dataset_name == 'PPI':
        g = dgl.data.PPIDataset(mode='train', verbose=False)[0]
        g.ndata['label'] = g.ndata['label'][:, 0].long()
        discrete_feat = True
    else:
        raise Exception('Dataset not implemented Error.')
    return g, discrete_feat


def cross_validation_gen(y, k_fold=5):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k_fold)
    train_splits = []
    val_splits = []
    test_splits = []

    for larger_group, smaller_group in skf.split(y, y):
        train_y = y[smaller_group]
        sub_skf = StratifiedKFold(n_splits=2)
        train_split, val_split = next(iter(sub_skf.split(train_y, train_y)))
        train = torch.zeros_like(y, dtype=torch.bool)
        train[smaller_group[train_split]] = True
        val = torch.zeros_like(y, dtype=torch.bool)
        val[smaller_group[val_split]] = True
        test = torch.zeros_like(y, dtype=torch.bool)
        test[larger_group] = True
        train_splits.append(train.unsqueeze(1))
        val_splits.append(val.unsqueeze(1))
        test_splits.append(test.unsqueeze(1))

    return torch.cat(train_splits, dim=1), torch.cat(val_splits, dim=1), torch.cat(test_splits, dim=1)
