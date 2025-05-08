import argparse
import copy
import logging

import torch
import wandb

from attack.gzoo import GZOO
from dataset.basic import Dataset
from models.utils import init_gnn_model, train_victim_model
from utils import seed_experiment

try:
    if 'logger' not in globals():
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
except NameError:
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def bool_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Graph Injection Attack')

    parser.add_argument('--seed', type=int, default=1027, help='Random seed')
    parser.add_argument('--device', type=int, default=0, help='Device ID for GPU')
    parser.add_argument('--dataset', type=str, default='citeseer', help='Dataset to attack')
    parser.add_argument('--victim_num_layers', type=int, default=2, help='Number of layers in victim models')
    parser.add_argument('--victim_model', type=str, default='gcn', help='Model architecture of the victim model')
    parser.add_argument('--hid_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Number of epochs to train')
    parser.add_argument('--load_pretrained', action='store_true')
    parser.add_argument('--save_dir', type=str, default='output', help='Save dir for injected nodes and edges')

    parser.add_argument('--attacker', type=str, default='GZoo', help='GIA algorithm')
    parser.add_argument('--node_budget', type=int, default=1, help='Node budget per node')
    parser.add_argument('--edge_budget', type=int, default=1, help='Edge budget')
    parser.add_argument('--feature_budget_factor', type=float, default=1, help='Feature budget multiplier, dummy for continuous case')

    # GZOO
    parser.add_argument('--gzoo_khop_edge', type=int, default=0, help='order of sub-graphs to wire node, 0 for full graph')
    parser.add_argument('--gzoo_kappa', type=float, default=-0.1, help='Margin of Loss')
    parser.add_argument('--gzoo_sigma', type=float, default=1e-6, help='Sample step size')
    parser.add_argument('--gzoo_eval_num', type=int, default=100, help='Number of evaluations')
    parser.add_argument('--gzoo_run_mode', type=str, default='black-box', help='black-box or white-box')
    parser.add_argument('--gzoo_discount_factor', type=float, default=0.95, help='discount factor')
    parser.add_argument('--gzoo_patience', type=int, default=4, help='Patience for early stopping')
    parser.add_argument('--gzoo_gen_feat_dim', type=int, default=32, help='Feature dim of generator')
    parser.add_argument('--gzoo_gen_hid_dim', type=int, default=64, help='Hidden dim of generator')
    parser.add_argument('--gzoo_gen_feat_mid_dim', type=int, default=4, help='Middle Hidden dim of generator')
    parser.add_argument('--gzoo_gen_lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gzoo_batch_size', type=int, default=0, help='batch-size for sub-dataset optimization')
    parser.add_argument('--gzoo_attack_epochs', type=int, default=5, help='attack epochs')
    parser.add_argument('--gzoo_alpha', type=float, default=1.0, help='balance factor')
    parser.add_argument('--gzoo_tau', type=float, default=1e-2, help='gradient scale factor')

    parser.add_argument('--instance', type=str, default='Attack', help='the instance name of wandb')
    parser.add_argument('--wandb_group', type=str, default='GZoo', help='the group name of wandb')

    parser.add_argument('--mismatch_mode', type=str, default='feature', help='msimatch')

    args = parser.parse_args()

    # froze random
    seed_experiment(seed=args.seed)
    device = 'cuda:{}'.format(str(args.device))

    dataset_name = args.dataset
    wandb_ins = wandb.init(
        project='Graph Injection Attack',
        name=args.instance,
        group=args.wandb_group,
        config={
            # basic config
            'seed': args.seed,
            'device': args.device,
            'dataset_name': args.dataset,
            'victim_num_layers': args.victim_num_layers,
            'victim_model': args.victim_model,
            'hid_dim': args.hid_dim,
            'epochs': args.epochs,
            'lr': args.lr,
            'load_pretrained': args.load_pretrained,
            'node_budget': args.node_budget,
            'feature_budget_factor': args.feature_budget_factor,
            'save_dir': args.save_dir,
            'attacker': args.attacker,

            # GZOO
            'gzoo_khop_edge': args.gzoo_khop_edge,
            'gzoo_kappa': args.gzoo_kappa,
            'gzoo_sigma': args.gzoo_sigma,
            'gzoo_eval_num': args.gzoo_eval_num,
            'gzoo_run_mode': args.gzoo_run_mode,
            'gzoo_discount_factor': args.gzoo_discount_factor,
            'gzoo_patience': args.gzoo_patience,
            'gzoo_gen_feat_dim': args.gzoo_gen_feat_dim,
            'gzoo_gen_hid_dim': args.gzoo_gen_hid_dim,
            'gzoo_gen_lr': args.gzoo_gen_lr,
            'gzoo_attack_epochs': args.gzoo_attack_epochs,
            'gzoo_batch_size': args.gzoo_batch_size,
            'gzoo_alpha': args.gzoo_alpha,

        }
    )
    # wandb_ins = None

    # loading dataset and init victim model
    checkpoint_dir = 'checkpoint'
    normalized = args.attacker in ['G2A2C', 'GZOO']
    dataset = Dataset(dataset_name=args.dataset, device=device, normalized=normalized)
    victim_model = init_gnn_model(args.victim_model, dataset.feature_dim, args.hid_dim, dataset.n_class, args.victim_num_layers, device)
    if not args.load_pretrained:
        victim_model = train_victim_model(victim_model, dataset.graph, lr=args.lr, epochs=args.epochs)
        ppath = r'{}/victim_model/{}_{}_checkpoint.pt'.format(checkpoint_dir, args.dataset, args.victim_model)
        state_dict = victim_model.state_dict()
        torch.save(state_dict, ppath)
    else:
        ppath = r'{}/victim_model/{}_{}_checkpoint.pt'.format(checkpoint_dir, args.dataset, args.victim_model)
        state_dict = torch.load(ppath, map_location={'cpu': device})
        victim_model.load_state_dict(state_dict)
    dataset.init_target_nodes(victim_model)

    node_num = dataset.target_nodes.shape[0]
    if node_num > 1000:
        dataset.target_nodes = dataset.target_nodes[torch.randint(0, node_num, size=(1000,), dtype=torch.long)]
    logger.info('Number of Target Nodes = {}'.format(len(dataset.target_nodes)))

    # GIA process
    attacker = None
    if args.attacker == 'GZOO':

        attacker = GZOO(
            dataset=dataset, khop_edge=args.gzoo_khop_edge, node_budget=args.node_budget, patience=args.gzoo_patience,
            edge_budget=args.edge_budget, kappa=args.gzoo_kappa, sigma=args.gzoo_sigma, eval_num=args.gzoo_eval_num,
            victim_model_name=args.victim_model, device=args.device, normalized=normalized, run_mode=args.gzoo_run_mode,
            feature_budget_factor=args.feature_budget_factor, gen_feat_dim=args.gzoo_gen_feat_dim, gen_hid_dim=args.gzoo_gen_hid_dim,
            gen_lr=args.gzoo_gen_lr, batch_size=args.gzoo_batch_size, attack_epochs=args.gzoo_attack_epochs, alpha=args.gzoo_alpha, tau=args.gzoo_tau,
            gen_feat_mid_dim=args.gzoo_gen_feat_mid_dim
        )
    else:
        raise NotImplementedError('Attacker {} is not implemented'.format(args.attacker))
    attacker.run_attack(victim_model, dataset.graph, dataset.target_nodes, wandb_ins=wandb_ins)
