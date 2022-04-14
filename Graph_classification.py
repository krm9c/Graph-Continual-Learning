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
    Total_loss=[]
    Gen_loss=[]
    For_loss=[]
    n_Tasks=dataset.num_classes
    for i in range(n_Tasks):
        print("The task number", i)
        train_loader, test_loader, mem_train_loader, mem_test_loader,\
            memory_train, memory_test = continuum_Graph_classification(dataset, memory_train, memory_test, batch_size=64, task_id=i)
        for epoch in range(1,epochs):
            Total,Gen,For=train_CL(model, criterion, optimizer, mem_train_loader, train_loader, task=i, graph=1, node=0, params=config)
            Total_loss.append(Total)
            Gen_loss.append(Gen)
            For_loss.append(For)
            # print(epoch, print_it)
            if epoch%print_it==0:
                # scheduler.step()
                train_acc = test_GC(model, train_loader)
                test_acc = test_GC(model, test_loader)
                mem_train_acc = test_GC(model, mem_train_loader)
                mem_test_acc = test_GC(model, mem_test_loader)
                accuracies_mem.append(mem_test_acc)
                accuracies_one.append(test_acc)
                print("#########################################################################")
                print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
                print(f'Mem Train Acc: {mem_train_acc:.4f}, Mem Test Acc: {mem_test_acc:.4f}')
                print("#########################################################################")
    import numpy as np
    del model, criterion, optimizer, memory_train, memory_test, Total_loss, Gen_loss, For_loss
    print(np.array(accuracies_one).reshape([-1]).shape, np.array(accuracies_mem).reshape([-1]).shape)
    return np.array(accuracies_one).reshape([-1]), np.array(accuracies_mem).reshape([-1])