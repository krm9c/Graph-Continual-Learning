import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv 
from torch_geometric.nn import global_mean_pool

from Lib import *
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

import sys 
sys.path.append('../')
sys.path.append('../../')
from Lib import *

def run(name_label, epochs, print_it, config, model,\
    criterion, optimizer, dataset):
    import random
    memory_train=[]
    memory_test=[]


    #import torch.optim.lr_scheduler as lrs
    #scheduler = lrs.ExponentialLR(optimizer, gamma=0.9)
    accuracies_mem = []
    accuracies_one=[]
    F1_mem = []
    F1_one=[]
    Total_loss=[]
    Gen_loss=[]
    For_loss=[]

    n_Tasks=dataset.num_classes
    x = dataset[0].x
    y = dataset[0].y
    edge_index = dataset[0].edge_index 
    continuum_data = continuum_node_classification(dataset, n_Tasks, num_classes=dataset.num_classes)
    # The arrays for data
    memory_train=[]
    memory_test=[]
    for id, task in enumerate(continuum_data):
        train_mask, _, test_mask = task
        # print("id", id, train_mask.sum(), test_mask.sum())
        memory_train.append(train_mask)
        memory_test.append(test_mask)
        for epoch in range(epochs):
            train_loader= Data(x=x,edge_index=edge_index, y=y, train_mask=train_mask)
            Total,Gen,For=train_CL( model, criterion, optimizer,\
                memory_train, train_loader, task=id, \
                graph = 0, node=1, params = { 'x_updates': config['x_updates'],\
                'theta_updates': config['theta_updates'],\
                'factor': config['factor'],\
                'x_lr': config['x_lr'],\
                'th_lr':config['th_lr'],\
                'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
                'batchsize':8, 'total_updates': config['total_updates']})
            
            Total_loss.append(Total)
            Gen_loss.append(Gen)
            For_loss.append(For)
            if epoch%print_it==0:
                # scheduler.step()
                # print(test_mask.shape)
                # print(train_mask.shape)
                # print(train_loader.x.shape, train_loader.y.shape, train_loader.edge_index.shape)
                test_acc, test_F1 = test_NC(model, train_loader, [test_mask])
                # print(test_acc, test_F1)
                mem_test_acc, mem_test_f1 = test_NC(model, train_loader, memory_test)
                # print(mem_test_acc, mem_test_f1)
                accuracies_mem.append(mem_test_acc)
                accuracies_one.append(test_acc)
                F1_mem.append(mem_test_f1)
                F1_one.append(test_F1)
                print(f'Task: {id:03d}, Epoch: {epoch:03d}, Test Acc: {test_acc:.3f},  Mem Test Acc: {mem_test_acc:.3f}, Test F1: {test_F1:.3f}, Mem Test F1: {mem_test_f1:.3f}')
    
    
    # The metrics from ER paper
    PM=test_F1
    if id>0:
        diff =[ abs(F1_mem[-1]-ele) for ele in F1_mem]
        # print(diff)
        # print(max(diff))
        FM=max(diff)
    else:
        FM=F1_mem[id]
    # The metric from catastrophic Forgetting paper
    AP=mem_test_acc
    if id>0:
        AF=abs(accuracies_mem[id]-accuracies_mem[id-1])
    else:
        AF=accuracies_mem[id]
    #After the task has been learnt
    print("##########################################")
    print(f'PM: {PM:.3f}, FM: {FM:.3f}, AP: {AP:.3f}, AF: {AF:.3f}')
    print("##########################################")
    import numpy as np
    F1_one=np.array(F1_one).reshape([-1])
    F1_mem=np.array(F1_mem).reshape([-1])
    accuracies_one = np.array(accuracies_one).reshape([-1])
    accuracies_mem=np.array(accuracies_mem).reshape([-1])
    print(accuracies_one.shape, accuracies_mem.shape, F1_one.shape, F1_mem.shape)
    del model, criterion, optimizer, memory_train, memory_test, Total_loss, Gen_loss, For_loss
    return accuracies_one, accuracies_mem,F1_one, F1_mem

