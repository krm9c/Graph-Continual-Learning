import ray
import json
from functools import partial
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch.nn import Linear

from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GraphConv, global_mean_pool
from torch_geometric.datasets import Planetoid
import collections


def train_CL(model, criterion, optimizer, mem_loader, train_loader, task, graph = 0, node = 1, \
            params = {'x_updates': 1,  'theta_updates':1, 'factor': 0.0001, 'x_lr': 0.0001,'th_lr':0.0001,\
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
            'batchsize':8, 'total_updates': 1000} ):
        device = params['device']
        def normalize_grad(input, p=2, dim=1, eps=1e-12):
            return input / input.norm(p, dim, True).clamp(min=eps).expand_as(input)
        import copy
        # We set up the iterators for the memory loader and the train loader
        if node <1:
            mem_iter = iter(mem_loader)
            task_iter = iter(train_loader)   
        # The main loop over all the batch
        for i in range(params['total_updates']): 
            if task>0:
                ##########################################
                ###########################################
                ## For GRAPH classification
                if graph==1:
                    ###########################################
                    ## this is for when graph or not graph
                    try:
                        data_t= next(task_iter)
                        if data_t.y.shape[0]<params['batchsize']:
                            task_iter = iter(train_loader)
                            data_t = next(task_iter)
                    except StopIteration:
                        task_iter = iter(train_loader)
                        data_t= next(task_iter)
                    # Extract a batch from the memory
                    try:
                        data_m= next(mem_iter)
                        if data_m.y.shape[0]<params['batchsize']:
                            mem_iter = iter(mem_loader)
                            data_m= next(mem_iter)    
                    except StopIteration:
                        mem_iter = iter(mem_loader)
                        data_m= next(mem_iter)
                    # Send the data to the device
                    data_m = data_m.to(device)
                    data_t = data_t.to(device)
                    # Apply the model on the task batch and the memory batch
                    out = model(data_t.x.float().to(device), data_t.edge_index.to(device), data_t.batch.to(device))  # Perform a single fo
                    out_m = model(data_m.x.float().to(device), data_m.edge_index.to(device), data_m.batch.to(device))
                    ## Get loss on the memory and task and put it together
                    J_P = criterion(out, data_t.y.to(device))
                    J_M = criterion(out_m, data_m.y.to(device))
                    ############## This is the J_x loss
                    #########################################################################################
                    # Add J_x  now
                    x_PN = copy.copy(data_m.x).to(device)
                    x_PN.requires_grad = True
                    epsilon = params['x_lr']
                    # The x loop
                    for epoch in range(params["x_updates"]):
                        crit = criterion(model(x_PN.float(), data_m.edge_index, data_m.batch), data_m.y)
                        loss = torch.mean(crit)
                        # Calculate the gradient
                        adv_grad = torch.autograd.grad( loss,x_PN)[0]
                        # Normalize the gradient values.
                        adv_grad = normalize_grad(adv_grad, p=2, dim=1, eps=1e-12)
                        x_PN = x_PN+ epsilon*adv_grad   
                    # The critical cost function
                    J_x_crit = (criterion(model(x_PN.float(), data_m.edge_index, data_m.batch), data_m.y))
                    ############### This is the loss J_th
                    #########################################################################################
                    cop = copy.deepcopy(model).to(device)
                    opt_buffer = torch.optim.Adam(cop.parameters(),lr = params['th_lr'])
                    J_PN_theta = criterion(model(data_m.x.float(), data_m.edge_index, data_m.batch), data_m.y)
                    for i in range(params["theta_updates"]):
                        opt_buffer.zero_grad()
                        loss_crit = criterion(cop(data_t.x.float(), data_t.edge_index, data_t.batch), data_t.y)
                        loss_m = torch.mean(loss_crit) 
                        loss_m.backward(retain_graph=True)
                        opt_buffer.step()
                    J_th_crit = criterion(cop(data_m.x.float(), data_m.edge_index, data_m.batch), data_m.y)
                    # Now, put together  the loss fully 
                    Total_loss= torch.mean(J_M+J_P)+ params['factor']*torch.mean(J_x_crit+J_th_crit)
                elif node ==1:
                    # print("I am doing node classification")
                    import random
                    rand_index = random.randint(0,len(mem_loader)-1)
                    # Send the data to the device
                    data_m = mem_loader[rand_index].to(device)
                    # Apply the model on the task batch and the memory batch
                    out = model(train_loader.x.to(device), train_loader.edge_index.to(device))  # Perform a single fo
                    ## Get loss on the memory and task and put it together
                    J_P = criterion(out[train_loader.train_mask], train_loader.y[train_loader.train_mask].to(device))
                    J_M = criterion(out[data_m], train_loader.y[data_m].to(device))
                    mem_mask=torch.logical_or(train_loader.train_mask, data_m)
                    
                    ############## This is the J_x loss
                    #########################################################################################
                    # Add J_x  now
                    x_PN = copy.copy(train_loader.x).to(device)
                    x_PN.requires_grad = True
                    epsilon = params['x_lr']
                    # The x loop
                    for epoch in range(params["x_updates"]):
                        crit = criterion(model(x_PN.to(device), train_loader.edge_index.to(device))[mem_mask],train_loader.y[mem_mask] )
                        loss = torch.mean(crit)
                        # Calculate the gradient
                        adv_grad = torch.autograd.grad( loss,x_PN)[0]
                        # Normalize the gradient values.
                        adv_grad = normalize_grad(adv_grad, p=2, dim=1, eps=1e-12)
                        x_PN = x_PN+ epsilon*adv_grad   
                    # The critical cost function
                    J_x_crit = criterion( model(x_PN.to(device), train_loader.edge_index.to(device))[mem_mask], train_loader.y[mem_mask] )
                    ############### This is the loss J_th
                    #########################################################################################
                    cop = copy.deepcopy(model).to(device)
                    opt_buffer = torch.optim.Adam(cop.parameters(),lr = params['th_lr'])
                    for i in range(params["theta_updates"]):
                        opt_buffer.zero_grad()
                        loss_crit = criterion(cop(train_loader.x.to(device),train_loader.edge_index)[mem_mask], train_loader.y[mem_mask])
                        loss_m = torch.mean(loss_crit) 
                        loss_m.backward(retain_graph=True)
                        opt_buffer.step()
                    J_th_crit = (criterion(cop(train_loader.x.to(device),train_loader.edge_index)[mem_mask], train_loader.y[mem_mask]))
                    # Now, put together  the loss fully 
                    Total_loss= torch.mean(J_M+J_P)+ params['factor']*torch.mean(J_x_crit-J_th_crit)

                ## For REGULAR NET
                else:
                    ###########################################
                    ## this is for when graph or not graph
                    try:
                        data_t= next(task_iter)
                        if data_t.y.shape[0]<params['batchsize']:
                            task_iter = iter(train_loader)
                            data_t = next(task_iter)
                    except StopIteration:
                        task_iter = iter(train_loader)
                        data_t= next(task_iter)
                    # Extract a batch from the memory
                    try:
                        data_m= next(mem_iter)
                        if data_m.y.shape[0]<params['batchsize']:
                            mem_iter = iter(mem_loader)
                            data_m= next(mem_iter)    
                    except StopIteration:
                        mem_iter = iter(mem_loader)
                        data_m= next(mem_iter)

                    in_t, targets_t= data_t
                    in_m, targets_m = data_m
                    in_t = in_t.unsqueeze(dim=1).float().to(device)
                    in_m = in_m.unsqueeze(dim=1).float().to(device)
                    targets_t=targets_t.to(device)
                    targets_m=targets_m.to(device)
                    out = model(in_t)
                    out_m = model(in_m)
                    ############## The task cost and the memory cost
                    #########################################################################################
                    J_P = criterion(out, targets_t.to(device))
                    J_M = criterion(out_m, targets_m.to(device))
                    ############## This is the J_x loss
                    #########################################################################################
                    x_PN = copy.copy(in_m).to(device)
                    x_PN.requires_grad = True
                    epsilon =params['x_lr']
                    for epoch in range(params["x_updates"]):
                        crit = criterion(model(x_PN.float() ), targets_m)
                        loss = torch.mean(crit)
                        adv_grad = torch.autograd.grad(loss,x_PN)[0]
                        # Normalize the gradient values.
                        adv_grad = normalize_grad(adv_grad, p=2, dim=1, eps=1e-12)
                        x_PN = x_PN+ epsilon*adv_grad
                    J_x_crit = (criterion(model(x_PN.float()), targets_m))
                    ############### This is the loss J_th
                    #########################################################################################
                    cop = copy.deepcopy(model).to(device)
                    opt_buffer = torch.optim.Adam(cop.parameters(),lr = params['th_lr'])
                    J_PN_theta = criterion(model(in_m.float()), targets_m)
                    for i in range(params["theta_updates"]):
                        opt_buffer.zero_grad()
                        loss_crit = criterion(cop(in_t.float()), targets_t)
                        loss_m = torch.mean(loss_crit) 
                        loss_m.backward(retain_graph=True)
                        opt_buffer.step()
                    J_th_crit = (criterion(cop(in_m.float()), targets_m))
                    Total_loss= torch.mean(J_M+J_P+params['factor']*J_x_crit+params['factor']*J_th_crit)
                
                
                optimizer.zero_grad()
                Total_loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                
                
                return Total_loss.detach().cpu(),\
                (torch.mean(J_M)+ torch.mean(params['factor']*J_x_crit)).detach().cpu(),(torch.mean(J_P)-torch.mean(params['factor']*J_th_crit)).detach().cpu()

            ## FOR when there is only one task
            else:
                if graph==1:
                    ###########################################
                    ## this is for when graph or not graph
                    try:
                        data_t= next(task_iter)
                    except StopIteration:
                        task_iter = iter(train_loader)
                        data_t= next(task_iter)
                    out = model(data_t.x.float().to(device), data_t.edge_index.to(device), data_t.batch.to(device))  # Perform a single forward pass.
                    critti= criterion(out, data_t.y.to(device))
                    Total_loss = torch.mean(critti)
                    optimizer.zero_grad() 
                    Total_loss.backward()  # Derive gradients.
                    optimizer.step()  # Update parameters based on gradients.
                elif node==1:
                    out = model(train_loader.x.to(device), train_loader.edge_index.to(device))  # Perform a single fo
                    ## Get loss on the memory and task and put it together
                    critti = criterion(out[train_loader.train_mask], train_loader.y[train_loader.train_mask].to(device))
                    Total_loss = torch.mean(critti)
                    optimizer.zero_grad() 
                    Total_loss.backward()  # Derive gradients.
                    optimizer.step()  # Update parameters based on gradients.
                else:
                    # Extract a batch from the task
                    try:
                        data_t= next(task_iter)
                    except StopIteration:
                        task_iter = iter(train_loader)
                        data_t= next(task_iter)
                    in_t, targets_t = data_t 
                    in_t = in_t.unsqueeze(dim=1).float()
                    critti= criterion(model(in_t.to(device)), targets_t.to(device))
                    Total_loss = torch.mean(critti)
                    optimizer.zero_grad()
                    Total_loss.backward()  # Derive gradients.
                    optimizer.step()  # Update parameters based on gradients. 
                return Total_loss.detach().cpu(), Total_loss.detach().cpu(), Total_loss.detach().cpu()



def continuum_node_classification( datas, n_Tasks, num_classes):
    dataset = datas[0]
    # print("features inside the start", dataset.x)
    n_labels = num_classes
    n_labels_per_task = n_labels//n_Tasks
    # print("lable, n_task", n_labels, n_labels_per_task)
    labels_of_tasks = {}
    tasks=[]
    for task_i in range(n_Tasks):
        labels_of_current_task = list(range( task_i * n_labels_per_task, (task_i+1) * n_labels_per_task ))
        conditions = torch.BoolTensor( [l in labels_of_current_task for l in dataset.y.detach().cpu()] )
        mask_of_task = torch.where(conditions, 
                                torch.tensor(1), 
                                torch.tensor(0) )        
        train_mask = (dataset.train_mask.to(torch.long) * mask_of_task).to(torch.bool)
        val_mask = (dataset.val_mask.to(torch.long) * mask_of_task).to(torch.bool)
        test_mask = (dataset.test_mask.to(torch.long) * mask_of_task).to(torch.bool)
        
        # print(train_mask.shape[0], val_mask.shape[0], test_mask.shape[0])
        tasks.append((train_mask, val_mask, test_mask))
    return tasks
                
            
def test_NC(model, x, edge, mask, y):
      model.eval()
      out = model(x, edge)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[mask] == y[mask]  # Check against ground-truth labels.
      print(int(test_correct.sum()), int(mask.sum()))
      test_acc = int(test_correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
      return test_acc


def test_GC(model, loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0            
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y.to(device)).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def continuum_Graph_classification(dataset, memory_train, memory_test, batch_size, task_id):
    import random
    # print("new task", task_id)
    stack = [(dataset[j].y==task_id).item() for j in range(len(dataset))]
    datas = [ dataset[k] for k,val in enumerate(stack) if val== True] 
    lengtha=len(datas)
    random.shuffle(datas)
    train_dataset = datas[:int(0.80*lengtha)]
    test_dataset = datas[int(0.80*lengtha):]
    memory_train+=train_dataset
    memory_test+=test_dataset
    # print(f'Number of training graphs: {len(train_dataset)}')
    # print(f'Number of test graphs: {len(test_dataset)}')    
    # print(f'Memory:  Number of training graphs: {len(memory_train)}')
    # print(f'Memory:  Number of test graphs: {len(memory_test)}')

    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    mem_train_loader = DataLoader(memory_train, batch_size=batch_size, shuffle=True)
    mem_test_loader = DataLoader(memory_test, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, mem_train_loader, mem_test_loader, memory_train, memory_test 



## The main code for the deephyper run...
def load_data(data_label):   
    import torch    
    if data_label == 'MUTAG' or data_label == 'ENZYMES' or data_label=='PROTEINS':
        from torch_geometric.datasets import TUDataset
        dataset = TUDataset(root='data/TUDataset', name=data_label)
        print()
        print(f'Dataset: {dataset}:')
        print('====================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return dataset
    elif data_label=='MNIST':
        from torch_geometric.datasets import GNNBenchmarkDataset
        dataset = GNNBenchmarkDataset(root='data/GNNBench', name='MNIST')
        print()
        print(f'Dataset: {dataset}:')
        print('====================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return dataset

    elif data_label=='Cora' or data_label=='PubMed' or data_label =='CiteSeer':
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import NormalizeFeatures

        dataset = Planetoid(root='data/Planetoid', name=data_label)
        data= dataset[0]
        print("from the load dataset", data.x)
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return dataset