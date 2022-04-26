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

from Lib import *
name_data='MUTAG'
def run(config: dict):
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import  GATConv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class GAT(torch.nn.Module):
        def __init__(self, nfeat, nclass, drop_rate, hidden, in_head, out_head):
            super(GAT, self).__init__()
            self.hid = hidden
            self.in_head = in_head
            self.out_head = out_head
            self.dropout=drop_rate
            self.conv1 = GATConv(nfeat, self.hid, heads=self.in_head, dropout=self.dropout)
            self.conv2 = GATConv(self.hid*self.in_head, nclass, concat=False,
                                heads=self.out_head, dropout=self.dropout)
        def forward(self, x, edge_index):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

     # The data characteristics
    dataset= load_data(name_data)
    n_Tasks=dataset.num_classes
    x = dataset[0].x
    y = dataset[0].y
    edge_index = dataset[0].edge_index 
    continuum_data = continuum_node_classification(dataset, n_Tasks, num_classes=dataset.num_classes)
    # The Gat model

    model = GAT(nfeat=dataset.num_node_features,\
                nclass=dataset.num_classes,\
                drop_rate=config['dropout'],\
                hidden=config['hidden'],\
                in_head=8,\
                out_head=1)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["decay"])
    # The arrays for data
    memory_train=[]
    memory_test=[]
    for id, task in enumerate(continuum_data):
        train_mask, _, test_mask = task
        memory_train.append(train_mask)
        memory_test.append(test_mask)
        for epoch in range(5000):
            train_loader= Data(x=x,edge_index=edge_index, y=y, train_mask=train_mask)
            _,_,_=train_CL( model, criterion, optimizer, memory_train, train_loader, task=id, \
                graph = 0, node=1, params = { 'x_updates': config['x_updates'],\
                                              'theta_updates': config['theta_updates'],\
                                              'factor': config['factor'],\
                                               'x_lr': config['x_lr'],\
                                               'th_lr':config['th_lr'],\
                                               'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
                                               'batchsize':8, 'total_updates': config['total_updates']})
        mem_test_acc= [ test_NC(model, train_loader.x, train_loader.edge_index, element, train_loader.y) for element in memory_test]
    return sum(mem_test_acc)/len(memory_test)


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
    default_config = {
        'hidden':8,\
        'decay':5e-04,\
        "dropout":0.5,\
        "learning_rate": 0.001,\
        'x_updates': 3,\
        'theta_updates':3, \
        'factor': 0.01,\
        'x_lr': 0.01,
        'th_lr': 0.01,
        'total_updates': 1000}

    from deephyper.problem import HpProblem
    problem = HpProblem()
    problem.add_hyperparameter((8, 128, "uniform"), "hidden")
    problem.add_hyperparameter((0.00001, 0.9, "log-uniform"), "decay")
    problem.add_hyperparameter((0.01, 1, "log-uniform"), "dropout")
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