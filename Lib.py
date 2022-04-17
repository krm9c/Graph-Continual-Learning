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
import higher

def _f1_node(model, x, edge, y, mask, d='cuda'):
    # print("1 In the accuracy calculateion")
    out = model(x.to(d), edge.to(d))
    pred=torch.argmax(out,1)
    from sklearn.metrics import f1_score
    f1_ = f1_score(y[mask].cpu().detach().numpy(), \
        pred[mask].cpu().detach().numpy(),average='micro' )
    # print("3 Out In the F1 calculation")
    return f1_

def plot_save(acc_m, acc_one, save_name, name_label, total_epoch, print_it, total_runs, nTasks):
    import numpy as np
    np.savetxt(save_name+name_label+'_acc_runs_task.csv', acc_one)
    np.savetxt(save_name+name_label+'_acc_runs_memory.csv', acc_m)
    mean_m=np.mean(acc_m, axis=0)
    yerr_m=np.std(acc_m, axis=0)
    mean_t=np.mean(acc_one, axis=0)
    yerr_t=np.std(acc_one, axis=0)
    x_lab=np.arange(mean_m.shape[0])*(print_it)


    print(mean_m.shape, yerr_m.shape)
    n_e_task=(total_epoch//print_it)
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D
    import seaborn as sns
    import matplotlib.font_manager as font_manager
    import itertools as iters
    from scipy.ndimage import gaussian_filter
    large = 14; med = 12; small = 9
    def cm2inch(value):
        return value/2.54
    plt.style.use('seaborn-white')
    COLOR = 'black'
    params = {'axes.titlesize': large,
              'legend.fontsize': med,
              'figure.figsize': (cm2inch(16),cm2inch(18)),
              'axes.labelsize': med,
              'axes.titlesize': small,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': small, 
              'font.family': "sans-serif",
              'font.sans-serif': "Myriad Hebrew",
                'text.color' : 'black',
                'axes.labelcolor' : COLOR,
                'axes.linewidth' : 0.3,
                'xtick.color' : 'black',
                'ytick.color' : 'black'}
    plt.rcParams.update(params)
    plt.rc('text', usetex = False)
    color =['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',\
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['mathtext.fontset'] = 'cm'
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots( 2,1, dpi = 1200 )
    palette = sns.color_palette("Greys", nTasks*10)
    color_idx = iters.cycle(palette)
    for i in range(nTasks):
        ax[0].add_patch( matplotlib.patches.Rectangle((i*n_e_task-1,0), n_e_task, 1.2, color=next(color_idx)) )
    palette = sns.color_palette("Greys", nTasks*10)
    color_idx = iters.cycle(palette)
    for i in range(nTasks):
        ax[1].add_patch( matplotlib.patches.Rectangle((i*n_e_task-1,0), n_e_task, 1.2, color=next(color_idx)) )

    palette = sns.color_palette("Set1", nTasks)
    color_idx = iters.cycle(palette)
    ## PLOT THINGS ABOUT THE memory
    curve=mean_m
    err=yerr_m
    fill_up = curve+err
    fill_down = curve-err
    ax[0].fill_between(x_lab, fill_up, fill_down, alpha=0.5, color=next(color_idx))
    ax[0].legend(loc='upper right')

    ## PLOT THINGS ABOUT THE task
    curve=mean_t
    err=yerr_t
    fill_up = curve+err
    fill_down = curve-err
    ax[1].fill_between(x_lab, fill_up, fill_down, alpha=0.5, color=next(color_idx))

    ax[1].legend(loc='upper right')
    ax[1].set_ylabel('Accuracy (Task)')
    ax[1].set_xlabel('Epochs $k$')
    ax[1].grid(True)
    ax[0].grid(True)
    ax[0].set_ylabel('Accuracy (Mem)')
    ax[0].set_xlabel('Epochs $k$')
    # ax[0].set_ylim([0.6, 1])
    # ax[1].set_ylim([0.6, 1])
    ax[0].set_xlim([0, ((total_epoch//print_it-1)*nTasks) ])
    ax[1].set_xlim([0, ((total_epoch//print_it-1)*nTasks) ])
    fig.tight_layout()
    plt.savefig(save_name+name_label+'_.png', dpi=300)

def normalize_grad(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, True).clamp(min=eps).expand_as(input)

def return_batch(loader, iterator, batch_size):
    try:
        data= next(iterator)
        if data.y.shape[0]<batch_size:
            task_iter = iter(loader)
            return next(iterator)
        return data
    except StopIteration:
        task_iter = iter(loader)
        return next(iterator)

def Graph_update(model, criterion, optimizer, mem_loader, train_loader, task, \
            params = {'x_updates': 1,  'theta_updates':1, 'factor': 0.0001, 'x_lr': 0.0001,'th_lr':0.0001,\
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
            'batchsize':8, 'total_updates': 1000} ):
        device=params['device']
        import copy
        mem_iter = iter(mem_loader)
        task_iter = iter(train_loader)  
        # The main loop over all the batch
        for i in range(params['total_updates']): 
            if task>0:
                ## this is for when graph or not graph
                data_t = return_batch(train_loader, task_iter, params['batchsize']).to(device)
                data_m = return_batch(mem_loader, mem_iter, params['batchsize']).to(device)
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
                opt_buffer = torch.optim.Adam(model.parameters(),lr = params['th_lr'])
                optimizer.zero_grad()
                with higher.innerloop_ctx(model, opt_buffer) as (fmodel, diffopt):
                    for _ in range(params["theta_updates"]):
                        loss_crit = criterion(fmodel(data_t.x.float(), data_t.edge_index, data_t.batch), data_t.y)
                        loss_m = torch.mean(loss_crit) 
                        diffopt.step(loss_m)
                    J_th_crit = torch.mean(criterion(fmodel(data_m.x.float(), data_m.edge_index, data_m.batch), data_m.y))
                    Total_loss= torch.mean(J_M+J_P)+ params['factor']*torch.mean(J_x_crit)+J_th_crit
                    Total_loss.backward() 
                optimizer.step() 
                return Total_loss.detach().cpu(),\
                    torch.mean(J_M+J_x_crit).detach().cpu(),\
                    torch.mean(J_P+J_th_crit).detach().cpu()

            else:
                data_t = return_batch(train_loader, task_iter, params['batchsize'])
                out = model(data_t.x.float().to(device), data_t.edge_index.to(device), data_t.batch.to(device))  # Perform a single forward pass.
                critti= criterion(out, data_t.y.to(device))
                Total_loss = torch.mean(critti)
                optimizer.zero_grad() 
                Total_loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
            return Total_loss.detach().cpu(), Total_loss.detach().cpu(), Total_loss.detach().cpu()

def Node_update(model, criterion, optimizer, mem_loader, train_loader, task, \
            params = {'x_updates': 1,  'theta_updates':1, 'factor': 0.0001, 'x_lr': 0.0001,'th_lr':0.0001,\
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
            'batchsize':8, 'total_updates': 1000} ):
        device=params['device']
        ## The main loop over all the batch
        for i in range(params['total_updates']): 
            if task>0:
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
                mem_mask=torch.logical_or(train_loader.train_mask.to(device), data_m.to(device))
                
                ############## This is the J_x loss
                #########################################################################################
                # Add J_x  now
                import copy
                x_PN = copy.copy(train_loader.x).to(device)
                x_PN.requires_grad = True
                epsilon = params['x_lr']
                # The x loop
                for epoch in range(params["x_updates"]):
                    crit = criterion(model(x_PN.to(device), train_loader.edge_index.to(device))[mem_mask],train_loader.y[mem_mask].to(device) )
                    loss = torch.mean(crit)
                    # Calculate the gradient
                    adv_grad = torch.autograd.grad( loss,x_PN)[0]
                    # Normalize the gradient values.
                    adv_grad = normalize_grad(adv_grad, p=2, dim=1, eps=1e-12)
                    x_PN = x_PN+ epsilon*adv_grad   
                # The critical cost function
                J_x_crit = criterion( model(x_PN.to(device), train_loader.edge_index.to(device))[mem_mask], train_loader.y[mem_mask].to(device) )
                # Derive gradients.
                # Update parameters based on gradients.
                ############### This is the loss J_th
                #########################################################################################
                opt_buffer = torch.optim.Adam(model.parameters(),lr = params['th_lr'])
                optimizer.zero_grad()
                with higher.innerloop_ctx(model, opt_buffer) as (fmodel, diffopt):
                    for _ in range(params["theta_updates"]):
                        loss_crit = criterion(fmodel(train_loader.x.to(device),train_loader.edge_index.to(device))[mem_mask], train_loader.y[mem_mask].to(device))
                        loss_m = torch.mean(loss_crit) 
                        diffopt.step(loss_m)
                    J_th_crit = (criterion(fmodel(train_loader.x.to(device),train_loader.edge_index.to(device))[mem_mask], train_loader.y[mem_mask].to(device)))
                    Total_loss=torch.mean(J_M+J_P)+ params['factor']*torch.mean(J_x_crit+J_th_crit)
                    Total_loss.backward() 
                optimizer.step() 
                return Total_loss.detach().cpu(),\
                    (torch.mean(J_M)+torch.mean(J_x_crit)).detach().cpu(),\
                    (torch.mean(J_P)+torch.mean(J_th_crit)).detach().cpu()

            else:
                out = model(train_loader.x.to(device), train_loader.edge_index.to(device))  # Perform a single fo
                ## Get loss on the memory and task and put it together
                critti = criterion(out[train_loader.train_mask], train_loader.y[train_loader.train_mask].to(device))
                Total_loss = torch.mean(critti)
                optimizer.zero_grad() 
                Total_loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                return Total_loss.detach().cpu(), Total_loss.detach().cpu(), Total_loss.detach().cpu()

def Reg_update(model, criterion, optimizer, mem_loader, train_loader, task, \
            params = {'x_updates': 1,  'theta_updates':1, 'factor': 0.0001, 'x_lr': 0.0001,'th_lr':0.0001,\
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
            'batchsize':8, 'total_updates': 1000} ):
        device=params['device']
        import copy
        mem_iter = iter(mem_loader)
        task_iter = iter(train_loader)  
        # The main loop over all the batch
        for i in range(params['total_updates']): 
            if task>0:
                ## this is for when graph or not graph
                data_t = return_batch(train_loader, task_iter, params['batchsize'])
                data_m = return_batch(mem_loader, mem_iter, params['batchsize'])
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

                ############## This is the J_x loss
                #########################################################################################
                opt_buffer = torch.optim.Adam(model.parameters(),lr = params['th_lr'])
                optimizer.zero_grad()
                with higher.innerloop_ctx(model, opt_buffer) as (fmodel, diffopt):
                    for _ in range(params["theta_updates"]):
                        loss_crit = criterion(fmodel(in_t), targets_t)
                        loss_m = torch.mean(loss_crit) 
                        diffopt.step(loss_m)
                    J_th_crit = torch.mean(criterion(fmodel(in_t), targets_t))
                    Total_loss= torch.mean(J_M+J_P)+ params['factor']*torch.mean(J_x_crit)+J_th_crit
                    Total_loss.backward() 
                optimizer.step() 

                return Total_loss.detach().cpu(),\
                    torch.mean(J_M+J_x_crit).detach().cpu(),\
                    torch.mean(J_P+J_th_crit).detach().cpu()
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

def train_CL(model, criterion, optimizer, mem_loader, train_loader, task, graph = 0, node = 1, \
            params = {'x_updates': 1,  'theta_updates':1, 'factor': 0.0001, 'x_lr': 0.0001,'th_lr':0.0001,\
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
            'batchsize':8, 'total_updates': 1000} ):
        device = params['device']
        if graph==1:
            return Graph_update(model, criterion, optimizer, mem_loader, train_loader, task, \
            params = {'x_updates': 1,  'theta_updates':1, 'factor': 0.0001, 'x_lr': 0.0001,'th_lr':0.0001,\
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
            'batchsize':8, 'total_updates': 1000} )
        elif node==1:
            return Node_update(model, criterion, optimizer, mem_loader, train_loader, task, \
            params = {'x_updates': 1,  'theta_updates':1, 'factor': 0.0001, 'x_lr': 0.0001,'th_lr':0.0001,\
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
            'batchsize':8, 'total_updates': 1000} )
        else:
            return Reg_update(model, criterion, optimizer, mem_loader, train_loader, task, \
            params = {'x_updates': 1,  'theta_updates':1, 'factor': 0.0001, 'x_lr': 0.0001,'th_lr':0.0001,\
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
            'batchsize':8, 'total_updates': 1000} )

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

def _Acc_node(model, x, edge, y, mask, d='cuda'):
    # print("1 In the accuracy calculateion")
    # print(x.shape, y.shape, edge.shape, mask.shape)
    model.eval()
    # print("2 In the accuracy calculateion")
    # print(x.shape, y.shape, edge.shape, mask.shape)
    out = model(x.to(d), edge.to(d))
    # print("3 In the accuracy calculateion")
    # print(x.shape, y.shape, edge.shape, mask.shape)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[mask] == y[mask].to(d)  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
    # print("4 Out the accuracy calculateion")
    # print(x.shape, y.shape, edge.shape, mask.shape)
    return test_acc

def test_NC(model, loader, masks, d="cuda"):
    model.eval()
    # print("begin")
    # print(loader.x.shape, loader.y.shape, loader.edge_index.shape)
    if len(masks) ==1:
        # print(masks[0].shape)
        acc=[_Acc_node(model, loader.x, loader.edge_index, loader.y, masks[0], d='cuda')]
        # print("came out now")
        f1=[_f1_node(model, loader.x, loader.edge_index, loader.y, masks[0], d='cuda')]
    else:
        # print(len(masks))
        acc=[_Acc_node(model, loader.x, loader.edge_index, loader.y, mask, d='cuda') for mask in masks]
        f1 =[_f1_node(model, loader.x, loader.edge_index, loader.y, mask, d='cuda') for mask in masks]
    # print("I am going out the test_NC")
    return sum(acc)/len(acc), sum(f1)/len(f1)
       
def test_GC(model, loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correct = 0   
    f1_=[]   
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y.to(device)).sum())  # Check against ground-truth labels.
        from sklearn.metrics import f1_score  
        f1_.append(f1_score(data.y.cpu().detach().numpy(),\
              pred.cpu().detach().numpy(),average='micro'))
    test_acc= (correct / len(loader.dataset))

    return test_acc, sum(f1_)/len(f1_)

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
        dataset = TUDataset(root='data/TUDataset', name=data_label).shuffle()
        print(f'Dataset: {dataset}:')
        print('====================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return dataset
    elif data_label=='MNIST':
        from torch_geometric.datasets import GNNBenchmarkDataset
        dataset = GNNBenchmarkDataset(root='data/GNNBench', name='MNIST').shuffle()
        print()
        print(f'Dataset: {dataset}:')
        print('====================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return dataset

    elif data_label=='Cora' or data_label=='PubMed' or data_label =='CiteSeer':
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root='data/Planetoid', name=data_label)
        data= dataset[0]
        print(data)
        print("from the load dataset", data.x)
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return dataset