import ray
import json
import pandas as pd
from functools import partial
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch import nn
from torch_geometric.loader import DataLoader

import torch 
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,  SAGEConv
from torch_geometric.nn import global_mean_pool

# from run_CL_graphs import *
import sys 
sys.path.append('../GCL/')
from Lib import *

def run(config:dict):
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--nhid', type=int, default=128, help='hidden size')
    parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
    parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
    parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
    parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
    parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
    args = parser.parse_args()


    return Run_it({'total_epoch': 500, 'print_it': 1000000, 'total_runs':1, 'decay':config['decay'],\
        'learning_Rate':config['lr'], 'hidden':int(config['hc']),\
        'dropout':config['dropout'], 'layers':int(config['n_lays']), 'x_updates': int(config['x_updates']),\
        'theta_updates': int(config['th_updates']), 'factor': 1, 'x_lr': 1e-07,'th_lr':1e-07,\
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
        'batchsize':16, 'total_updates': config['tot_updates'],\
        'name_label':'cora_ML', 'save_dir':'Results/',\
        'prob':'node_class','model_parse':args, 'full':0,\
        'model_tit': 'GAT', 'num_labels_task':2})


def get_evaluator(run_function):
    from deephyper.evaluator.callback import LoggerCallback
    is_gpu_available = torch.cuda.is_available()
    import os
    print("GPU is available? ", is_gpu_available, os.environ.get("CUDA_VISIBLE_DEVICES"))
    # Default arguments for Ray: 1 worker and 1 worker per evaluation
    method_kwargs = {
        "address":'auto',
        "num_cpus_per_task": 1,
        "callbacks": [LoggerCallback()]
    }
    # If GPU devices are detected then it will create 'n_gpus' workers
    # and use 1 worker for each evaluation
    if is_gpu_available:
        method_kwargs["num_gpus_per_task"] = 1
    evaluator = Evaluator.create(
        run_function,
        method="subprocess",
        method_kwargs={"num_workers": 2},
    )
    print(f"Created new evaluator with \
        {evaluator.num_workers} worker{'s' if evaluator.num_workers > 1 else ''} \
        and config: {method_kwargs}")
    return evaluator


def problem():
    # We define a dictionnary for the default values
    default_config={
        'decay':1e-04,\
        'lr':1e-04,\
        'hc':16,\
        'dropout':0.6,\
        'n_lays':10,\
        'x_updates': 2,\
        'th_updates':5,
        'tot_updates': 500}

    from deephyper.problem import HpProblem
    problem = HpProblem()
    problem.add_hyperparameter((1e-06, 1, "log-uniform"), "decay")
    problem.add_hyperparameter((1e-05, 0.1, "log-uniform"), "lr")
    problem.add_hyperparameter((0.01, 1, "uniform"), "dropout")
    problem.add_hyperparameter((8, 128, "uniform"), "hc")
    problem.add_hyperparameter((1, 20, "uniform"), "n_lays")
    problem.add_hyperparameter((1, 20, "uniform"), "x_updates")
    problem.add_hyperparameter((2, 20, "uniform"), "th_updates")
    problem.add_hyperparameter((50, 5000, "uniform"), "tot_updates")
    
    # Add a starting point to try first
    problem.add_starting_point(**default_config)
    return problem

if __name__ == "__main__":
    print("This the deephyper stuff")
    from deephyper.search.hps import AMBS
    from deephyper.evaluator import Evaluator
    prob1 = problem()
    evaluator_1= get_evaluator(run)
    print("the total number of deep hyper workers are", evaluator_1.num_workers) 
    # Instanciate the search with the problem and a specific evaluator
    search = AMBS(prob1, evaluator_1)
    results = search.search(max_evals=100)
    print("The deephyper results are", results)
    np.savetxt('results_fi/filename_citeseer.csv', results)