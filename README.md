
## Abstract

Graph Neural Networks (GNNs) have been extensively applied in critical domains, making the assessment of GNN robustness an urgent requirement.
However, current node injection attacks suffer from two primary issues.
Firstly, they generate attributes and structures for malicious nodes independently, overlooking the interrelationship between attributes and structures.
Secondly, these attacks usually involve training surrogate models to mimic the victim model, potentially causing significant performance degradation if the surrogate model's architecture does not precisely match the victim model.
To overcome these limitations, we propose a novel black-box node injection attack method, namely GZOO, which effectively degrades the performance of various GNNs.
GZOO leverages an adversarial graph generative model to simultaneously fabricate attributes and structures for malicious nodes.
Furthermore, we devise an algorithm based on zeroth-order optimization to update the generative model's parameters, thereby eliminating the need for surrogate model training.
Through comprehensive evaluations across eleven benchmark datasets, we demonstrate that GZOO outperforms existing state-of-the-art attacks in terms of effectiveness, robustness, and flexibility.

## Preparations

### Conda Environment Installation

1. First, please install [miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe) or [anaconda](https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Windows-x86_64.exe)

2. Using the following command create a new environment, named as ```secgnn```
```Bash
conda env create -f secgnn.yaml
```

### Init Logger

This repo utilizes the ```wandb``` package to log the training process.
Thus, please following the instructions of [its Official GitHub Repo](https://github.com/wandb/wandb/tree/main) to configure the ```wandb```.

Note that, after configuring it, please update ```project name``` in the line 138 of ```main.py```.

## Running

### Effectiveness

1. Victim model is "GCN" and Attack algorithm is "GZOO"
```shell
python main.py --seed 1027 --dataset=cora --node_budget=1 --edge_budget=1 --feature_budget=1 --victim_num_layers=2 --victim_model=sgc --hid_dim=256 --lr=0.001 --attacker=GZOO --gzoo_khop_edge=2 --gzoo_kappa=-0.001 --gzoo_run_mode=black-box --gzoo_attack_epochs=5 --gzoo_sigma=1e-3 --gzoo_patience=4 --gzoo_gen_feat_dim=64 --gzoo_gen_hid_dim=128 --gzoo_gen_lr=1e-3 --gzoo_batch_size=64 --instance=GZOO-cora-SGC-1027 --wandb_group=Effectiveness
```

More examples please refer to ```effectiveness.sh```

### Robustness

1. Victim model is "RobustGCN" and Attack algorithm is "GZOO"

```shell
python main.py --seed 1027 --dataset=cora --node_budget=1 --edge_budget=1 --feature_budget=1 --victim_num_layers=2 --victim_model=rgat --hid_dim=256 --lr=0.001 --attacker=GZOO --gzoo_khop_edge=2 --gzoo_kappa=-0.001 --gzoo_run_mode=black-box --gzoo_attack_epochs=5 --gzoo_sigma=1e-3 --gzoo_patience=40 --gzoo_gen_feat_dim=64 --gzoo_gen_hid_dim=128 --gzoo_gen_lr=1e-3 --gzoo_batch_size=64 --instance=GZOO-cora-RGAT-1027 --wandb_group=Robustness
```


### Flexibility

1. Victim model is "SGC" and Dataset is "CoauthorCS"

```shell
python main.py --seed 1027 --dataset=CoauthorCS --node_budget=1 --edge_budget=1 --feature_budget=1 --victim_num_layers=2 --victim_model=sgc --hid_dim=256 --lr=0.001 --attacker=GZOO --gzoo_khop_edge=2 --gzoo_kappa=-0.001 --gzoo_run_mode=black-box --gzoo_attack_epochs=5 --gzoo_sigma=1e-3 --gzoo_patience=4 --gzoo_gen_feat_dim=64 --gzoo_gen_hid_dim=128 --gzoo_eval_num=50 --gzoo_gen_lr=1e-3 --gzoo_batch_size=64 --instance=GZOO-CoauthorCS-SGC-1027 --wandb_group=Generalization
```

2. Victim model is "GCN", Dataset is "co_computer" and Node budget is "2"

```shell
python main.py --seed 1027 --dataset=co_computer --node_budget=2 --edge_budget=1 --feature_budget=1 --victim_num_layers=2 --victim_model=gcn --hid_dim=256 --lr=0.001 --attacker=GZOO --gzoo_khop_edge=2 --gzoo_kappa=-0.001  --gzoo_run_mode=black-box --gzoo_attack_epochs=5 --gzoo_sigma=1e-3 --gzoo_patience=4 --gzoo_gen_feat_dim=64 --gzoo_gen_hid_dim=128 --gzoo_gen_lr=1e-3 --gzoo_batch_size=64 --instance=GZOO-co_computer-GCN-Node-2-1027 --wandb_group=Generalization
```

### Citation

```biblatex
@article{yu2024zoo,
  title={GZOO: Black-box Node Injection Attack on Graph Neural Networks via Zeroth-order Optimization},
  author={Yu, Hao and Liang, Ke and Hu, Dayu and Tu, Wenxuan and Ma, Chuan and Zhou, Sihang and Liu, Xinwang},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024},
  publisher={IEEE}
}
```