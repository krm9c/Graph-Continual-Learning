from configparser import LegacyInterpolation
from locale import PM_STR
from plistlib import FMT_BINARY
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
    for i in range(n_Tasks):
        # print("The task number", i)
        train_loader, test_loader, mem_train_loader, mem_test_loader,\
            memory_train, memory_test = continuum_Graph_classification(dataset, memory_train, memory_test, batch_size=64, task_id=i)
        for epoch in range(1,epochs):
            Total,Gen,For=train_CL(model, criterion, optimizer, mem_train_loader, train_loader, task=i, graph=1, node=0, params=config)
            Total_loss.append(Total)
            Gen_loss.append(Gen)
            For_loss.append(For)
            if epoch%print_it==0:
                # scheduler.step()
                train_acc, train_F1 = test_GC(model, train_loader)
                test_acc, test_F1 = test_GC(model, test_loader)
                mem_train_acc, mem_train_f1 = test_GC(model, mem_train_loader)
                mem_test_acc, mem_test_f1 = test_GC(model, mem_test_loader)
                # print(test_F1, mem_test_f1)
                accuracies_mem.append(mem_test_acc)
                accuracies_one.append(test_acc)
                F1_mem.append(mem_test_f1)
                F1_one.append(test_F1)
                print(f'Task: {i:03d}, Epoch: {epoch:03d}, Test Acc: {test_acc:.3f},  Mem Test Acc: {mem_test_acc:.3f}, Test F1: {test_F1:.3f}, Mem Test F1: {mem_test_f1:.3f}')
    
    
    # The metrics from ER paper
    PM=test_F1
    if i>0:
        diff =[ abs(F1_mem[-1]-ele) for ele in F1_mem]
        print(diff)
        print(max(diff))
        FM=max(diff)
    else:
        FM=F1_mem[i]
    # The metric from catastrophic Forgetting paper
    AP=mem_test_acc
    if i>0:
        AF=abs(accuracies_mem[i]-accuracies_mem[i-1])
    else:
        AF=accuracies_mem[i]
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