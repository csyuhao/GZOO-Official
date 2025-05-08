import copy
import logging
from itertools import chain

import torch
import torch.nn.functional as F

from defense.gnnguard import GCNGuard
from defense.gtrans import GTrans
from defense.magnet import MAGNet
from defense.prognn import ProGNN
from defense.rgat import RGAT
from models.appnp import APPNP
from models.chebnet import ChebNet
from models.gat import GAT
from models.gcn import GCN
from models.pcnet import PCNet
from models.sgc import SGC
from defense.rgcn import RobustGCN
from models.sgformer import SGFormer

try:
    if 'logger' not in globals():
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
except NameError:
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def init_gnn_model(model_name, feat_dim, hid_dim, n_class, num_layers, device):
    if model_name == 'gcn':
        model = GCN(feat_dim, hid_dim, n_class, num_layers).to(device)
    elif model_name == 'gat':
        model = GAT(feat_dim, hid_dim, n_class, num_layers).to(device)
    elif model_name == 'sgc':
        model = SGC(feat_dim, hid_dim, n_class, num_layers).to(device)
    elif model_name == 'appnp':
        model = APPNP(feat_dim, hid_dim, n_class).to(device)
    elif model_name == 'chebnet':
        model = ChebNet(feat_dim, hid_dim, n_class).to(device)
    elif model_name == 'pcnet':
        model = PCNet(feat_dim, hid_dim, n_class, num_layers, n_poly=10, alpha=0.5, a=0.0, b=1.3, c=0.45).to(device)
    elif model_name == 'sgformer':
        model = SGFormer(feat_dim, hid_dim, n_class).to(device)
    elif model_name == 'rgcn':
        model = RobustGCN(feat_dim, n_class, n_hids=[hid_dim] * num_layers).to(device)
    elif model_name == 'rgat':
        model = RGAT(feat_dim, hid_dim, n_class, num_layers, dropout=0.5).to(device)
    elif model_name == 'gcnguard':
        model = GCNGuard(feat_dim, n_class, hids=[hid_dim] * num_layers).to(device)
    elif model_name == 'prognn':
        model = ProGNN(feat_dim, hid_dim, n_class, device=device).to(device)
    elif model_name == 'gtrans':
        model = GTrans(feat_dim, hid_dim, n_class).to(device)
    elif model_name == 'magnet':
        model = MAGNet(feat_dim, hid_dim, n_class, device=device).to(device)
    else:
        raise Exception('Model not implemented err.')
    return model


def train_victim_model(model, g, lr=0.01, epochs=1000):
    """ Training Graph Neural Networks
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    test_idx = None
    best_state_dict = None
    for e in range(epochs):

        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            # best_model = copy.deepcopy(model.detach())
            best_state_dict = copy.deepcopy(model.state_dict())
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logging.info('Victim model has validation accuracy: {:.2f}, testing accuracy: {:.2f}'.format(
        best_val_acc.item() * 100, best_test_acc.item() * 100
    ))
    model.load_state_dict(best_state_dict)
    return model


def train_surrogate_model(model, g, lr=0.01, epochs=1000):
    """ Training Graph Neural Networks (Using validation nodes)
    Assuming black-box attack
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    test_idx = None
    best_state_dict = None
    for e in range(epochs):

        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[val_mask], labels[val_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_state_dict = copy.deepcopy(model.state_dict())
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logging.info('Surrogate model has validation accuracy: {:.2f}, testing accuracy: {:.2f}'.format(
        best_val_acc.item() * 100, best_test_acc.item() * 100
    ))
    model.load_state_dict(best_state_dict)
    return model
