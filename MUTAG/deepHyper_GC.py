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

from multi_run import *
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
    default_config = {'decay':5e-4,'learning_Rate':0.01,\
                    'hidden_channels':64, 'dropout':0.6, 'layers':2,
                    'x_updates': 1,  'theta_updates':1, 'factor': 0.0001, 'x_lr': 0.0001,'th_lr':0.0001,\
                    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
                    'batchsize':64, 'total_updates': 2000}

    from deephyper.problem import HpProbålem
    problem = HpProblem()
    problem.add_hyperparameter((1, 4, "uniform"), "hiddenlayer")
    problem.add_hyperparameter((8, 128, "uniform"), "batch_size")
    problem.add_hyperparameter((0.01, 1, "log-uniform"), "dropout")
    problem.add_hyperparameter((8, 128, "uniform"), "hidden")
    problem.add_hyperparameter((0, 2, "uniform"), "layer_type")
    problem.add_hyperparameter((0.001, 0.1, "log-uniform"), "learning_rate")
    


    ## The hyper-parameters for the continual learning part.
    problem.add_hyperparameter((1, 10, "uniform"), "x_updates")
    problem.add_hyperparameter((1, 10, "uniform"), "theta_updates")
    problem.add_hyperparameter((0.000001, 0.1, "uniform"), "factor")
    problem.add_hyperparameter((0.0001, 0.1, "log-uniform"), "x_lr")
    problem.add_hyperparameter((0.0001, 0.1, "log-uniform"), "th_lr")
    problem.add_hyperparameter((50, 1200, "uniform"), "total_updates")
    # Add a starting point to try first
    problem.add_starting_point(**default_config)
    return problem



if __name__ == "__main__":
    from deephyper.search.hps import AMBS
    from deephyper.evaluator import Evaluator
    prob1 = problem()
    evaluator_1= get_evaluator(Run_it)
    print("the total number of deep hyper workers are", evaluator_1.num_workers) 
    # Instanciate the search with the problem and a specific evaluator
    search = AMBS(prob1, evaluator_1)
    search.search(max_evals=20)