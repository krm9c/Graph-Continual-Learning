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


from run_CL import *
def run(config:dict):
    Run_it({'total_epoch': 5000, 'print_it': 1000000, 'total_runs':1, 'decay':config['decay'],\
        'learning_Rate':config['lr'],\
        'hidden_channels':config['hc'], 'dropout':config['dropout'], 'layers':config['n_lays'],\
        'x_updates': config['x_updates'],  'theta_updates':config['th_updates'],\
        'factor': 1, 'x_lr': 1e-06,'th_lr':1e-04,\
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
        'batchsize':16, 'total_updates': config['tot_updates'], 'name_label':'PROTEINS', 'save_dir':'Results/proteins/',\
        'prob':'graph_class','model_parse':args, 'full':0, 'model_tit': 'GCN' })

def get_evaluator(run_function):
    from deephyper.evaluator.callback import LoggerCallback
    is_gpu_available = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count()
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
        method="ray",
        method_kwargs=method_kwargs
    )
    # print(f"Created new evaluator with {evaluator.num_workers} worker{'s' if evaluator.num_workers > 1 else ''} and config: {method_kwargs}", )
    return evaluator


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
        method="ray",
        method_kwargs=method_kwargs
    )
    print(f"Created new evaluator with {evaluator.num_workers} worker{'s' if evaluator.num_workers > 1 else ''} and config: {method_kwargs}", )
    return evaluator


def problem():
    # We define a dictionnary for the default values
    default_config=Run_it({
        'decay':1e-04,\
        'lr':1e-04,\
        'hc':16,\
        'dropout':0.6,\
        'n_lays':10,\
        'x_updates': 2,\
        'th_updates':5,
        'tot_updates': 500})

    from deephyper.problem import HpProbålem
    problem = HpProblem()
    problem.add_hyperparameter((1, 4, "uniform"), "decay")
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
    from deephyper.search.hps import AMBS
    from deephyper.evaluator import Evaluator
    prob1 = problem()
    evaluator_1= get_evaluator(run)
    print("the total number of deep hyper workers are", evaluator_1.num_workers) 
    # Instanciate the search with the problem and a specific evaluator
    search = AMBS(prob1, evaluator_1)
    search.search(max_evals=20)