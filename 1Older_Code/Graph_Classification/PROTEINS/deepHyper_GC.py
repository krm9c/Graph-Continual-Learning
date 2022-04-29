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

from GCL.Lib import *

name_label='MUTAG'
def run(config: dict):
    import torch
    from torch.nn import Linear
    import torch.nn.functional as F
    from torch_geometric.nn import  SAGEConv, GCNConv, GraphConv
    from torch_geometric.nn import global_mean_pool

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels, drop_rate, hidden, layer_index, n_inputs, n_classes):
            super(GCN, self).__init__()
            layers = [SAGEConv, GCNConv, GraphConv]
            self.conv1 = layers[layer_index](n_inputs, hidden_channels).to(device)
            self.conv2=[]
            for i in range(hidden):
                self.conv2.append(layers[layer_index](hidden_channels, hidden_channels).to(device) )
            self.conv3 = layers[layer_index](hidden_channels, hidden_channels).to(device)
            self.drop=drop_rate
            self.lins=[]
            for i in range(hidden):
                self.lins.append( Linear(hidden_channels, hidden_channels).to(device) )
            self.lin = Linear(hidden_channels, n_classes).to(device)

        def forward(self, x, edge_index, batch):
            # 1. Obtain node embeddings 
            x = self.conv1(x, edge_index)
            x = x.relu()

            for i in range(len(self.conv2)):
                x = self.conv2[i](x, edge_index)
                x = x.relu()

            x = self.conv3(x, edge_index)
            # 2. Readout layer
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
            # 3. Apply a final classifier
            for i in range(len(self.lins)):
                x = self.lins[i](x)
                x = x.relu()
            x = F.dropout(x, p=self.drop, training=self.training)
            x = self.lin(x)
            return x

    dataset = load_data(name_label)
    model = GCN(hidden_channels=int(config['hidden']),\
    hidden_channels=int(config['hiddenlayer']),\
    drop_rate=float(config['dropout']),\
    hidden=int(config['hidden']),\
    layer_index=int(config['layer_type']),\
    n_inputs=int(dataset.num_features),\
    num_classes=int(dataset.num_classes))

    n_Tasks=dataset.num_classes
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    for i in range(n_Tasks):
        print("The task number", i)
        train_loader, test_loader, mem_train_loader, mem_test_loader,\
            memory_train, memory_test = continuum_Graph_classification(dataset, memory_train, memory_test, batch_size=64, task_id=i)
        for epoch in range(1,50):
            _,_,_=train_CL( model, criterion, optimizer, memory_train, train_loader, task=id, \
                graph = 1, node=0, params = { 'x_updates': int(config['x_updates']),\
                                              'theta_updates': int(config['theta_updates']),\
                                              'factor': float(config['factor']),\
                                               'x_lr': float(config['x_lr']),\
                                               'th_lr':float(config['th_lr']),\
                                               'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
                                               'batchsize':int(config['batch_size']),\
                                                'total_updates': int(config['total_updates'])})
            
    return test_GC(model, mem_test_loader)


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
    default_config ={"hiddenlayer": 2,\
        "batch_size":64,\
        "dropout":0.5,\
        "hidden":32,  "layer_type":2,\
        "learning_rate": 0.001,\
        "x_updates": 3,\
        "theta_updates":3, \
        "factor": 0.01,\
        "x_lr": 0.01,\
        "th_lr": 0.01,\
        "total_updates": 1000}

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
    evaluator_1= get_evaluator(run)
    print("the total number of deep hyper workers are", evaluator_1.num_workers) 
    # Instanciate the search with the problem and a specific evaluator
    search = AMBS(prob1, evaluator_1)
    search.search(max_evals=20)