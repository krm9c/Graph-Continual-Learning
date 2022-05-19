import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import higher
import numpy as np
from sklearn.metrics import f1_score
import copy
from torch.profiler import profile, record_function, ProfilerActivity
import random
from model import *
import collections
from queue import Empty
import sys 

# Set some macros for reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def plot_save(acc_m, acc_one, save_name, name_label, total_epoch, print_it, total_runs, nTasks):
    np.savetxt(save_name+name_label+'_acc_runs_task.csv', acc_one)
    np.savetxt(save_name+name_label+'_acc_runs_memory.csv', acc_m)
    mean_m=np.mean(acc_m, axis=0)
    yerr_m=np.std(acc_m, axis=0)
    mean_t=np.mean(acc_one, axis=0)
    yerr_t=np.std(acc_one, axis=0)
    x_lab=np.arange(mean_m.shape[0])
    # print(mean_m.shape, yerr_m.shape)
    n_e_task=(total_epoch//print_it)
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import itertools as iters
    large = 20; med = 18; small = 16
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
    # ax[0].legend(loc='upper right')
    ## PLOT THINGS ABOUT THE task
    curve=mean_t
    err=yerr_t
    fill_up = curve+err
    fill_down = curve-err
    ax[1].fill_between(x_lab, fill_up, fill_down, alpha=0.5, color=next(color_idx))
    # ax[1].legend(loc='upper right')
    ax[1].set_ylabel('Accuracy (Task)')
    ax[1].set_xlabel('Tasks $k$')
    ax[1].grid(True)
    ax[0].grid(True)
    ax[0].set_ylabel('Accuracy (Mem)')
    ax[0].set_xlabel('Tasks $k$')
    # ax[0].set_ylim([0.6, 1])
    # ax[1].set_ylim([0.6, 1])
    ax[0].set_xlim([0, nTasks])
    ax[1].set_xlim([0, nTasks])
    fig.tight_layout()
    plt.savefig(save_name+name_label+'_.png', dpi=300)
    plt.close()



def plot_save_loss(Total_loss, Gen_loss, For_loss, save_name, name_label):
    print(Total_loss.shape, Gen_loss.shape, For_loss.shape)
    np.savetxt(save_name+name_label+'_total.csv', Total_loss)
    np.savetxt(save_name+name_label+'_for.csv', Gen_loss)
    np.savetxt(save_name+name_label+'_for.csv', For_loss)


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
        mem_iter = iter(mem_loader)
        task_iter = iter(train_loader)  
        # The main loop over all the batch
        for i in range(params['total_updates']): 
            if task>0:
                ## this is for when graph or not graph
                data_t = return_batch(train_loader, task_iter, params['batchsize'])
                data_m = return_batch(mem_loader, mem_iter, params['batchsize'])
                x = data_t.x.float().to(device)
                y= data_t.y.to(device)
                edge_index= data_t.edge_index.to(device)
                batch=data_t.batch.to(device)
                
                x_m = data_m.x.float().to(device)
                y_m= data_m.y.to(device)
                edge_index_m= data_m.edge_index.to(device)
                batch_m=data_m.batch.to(device)

                # Apply the model on the task batch and the memory batch
                out = model(x, edge_index, batch)  # Perform a single fo
                out_m = model(x_m, edge_index_m, batch_m)
                ## Get loss on the memory and task and put it together
                J_P = criterion(out,y )
                J_M = criterion(out_m, y_m)
                ############## This is the J_x loss
                #########################################################################################
                # Add J_x  now
                x_PN = copy.copy(data_m.x).to(device)
                x_PN.requires_grad = True
                epsilon = params['x_lr']
                # The x loop
                for _ in range(params["x_updates"]):
                    crit = criterion(model(x_PN.float(), edge_index_m, batch_m), y_m)
                    loss = torch.mean(crit)
                    # Calculate the gradient
                    adv_grad = torch.autograd.grad(loss,x_PN)[0]
                    # Normalize the gradient values.
                    adv_grad = normalize_grad(adv_grad, p=2, dim=1, eps=1e-12)
                    x_PN = x_PN+ epsilon*adv_grad   

                # The critical cost function
                J_x_crit = (criterion(model(x_PN.float(), edge_index_m, batch_m), y_m))
                ############### This is the loss J_th
                #########################################################################################
                optimizer.zero_grad()
                opt_buffer = torch.optim.Adam(model.parameters(),lr = params['th_lr'])
                with higher.innerloop_ctx(model, opt_buffer) as (fmodel, diffopt):
                    for _ in range(params["theta_updates"]):
                        loss_crit = criterion(fmodel(x, edge_index, batch), y)
                        loss_m = torch.mean(loss_crit) 
                        diffopt.step(-1*loss_m)
                    J_th_crit = torch.mean(criterion(fmodel(x, edge_index, batch), y))
                    Total_loss= torch.mean(J_M+J_P)+ params['factor']*torch.mean(J_x_crit+J_th_crit)
                    Total_loss.backward()
                optimizer.step()
                return Total_loss.detach().cpu(), torch.mean(J_M+J_x_crit).detach().cpu(), torch.mean(J_P+J_th_crit).detach().cpu()
            else:
                data_t = return_batch(train_loader, task_iter, params['batchsize'])
                x = data_t.x.float().to(device)
                y= data_t.y.to(device)
                edge_index= data_t.edge_index.to(device)
                batch=data_t.batch.to(device)
                out = model(x, edge_index, batch)  # Perform a single fo
                Total_loss = torch.mean(criterion(out,y))
                optimizer.zero_grad() 
                Total_loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                return Total_loss.detach().cpu(), Total_loss.detach().cpu(), Total_loss.detach().cpu()

# def Node_update_CCC(model, criterion, optimizer, mem_loader, train_loader, task, \
#             params = {'x_updates': 1,  'theta_updates':1, 'factor': 0.0001,\
#                  'x_lr': 0.0001,'th_lr':0.0001,\
#             'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
#             'batchsize':8, 'total_updates': 1000} ):
#         device=params['device']
#         for i in range(params['total_updates']): 
#             if task>0:
#                 # print("I am doing node classification")
#                 import random
#                 rand_index = random.randint(0,len(mem_loader)-1)
#                 # Send the data to the device
#                 data_m = mem_loader[rand_index].to(device)
#                 # Apply the model on the task batch and the memory batch
#                 out = model(train_loader.x.to(device), train_loader.edge_index.to(device))  # Perform a single fo
#                 ## Get loss on the memory and task and put it together
#                 J_P = criterion(out[train_loader.train_mask], train_loader.y[train_loader.train_mask].to(device))
#                 J_M = criterion(out[data_m], train_loader.y[data_m].to(device))
#                 mem_mask=torch.logical_or(train_loader.train_mask.to(device), data_m.to(device))
                
#                 ############## This is the J_x loss
#                 #########################################################################################
#                 # Add J_x  now
#                 import copy
#                 x_PN = copy.copy(train_loader.x).to(device)
#                 x_PN.requires_grad = True
#                 epsilon = params['x_lr']
#                 # The x loop
#                 for epoch in range(params["x_updates"]):
#                     crit = criterion(model(x_PN.to(device), train_loader.edge_index.to(device))[mem_mask],train_loader.y[mem_mask].to(device) )
#                     loss = torch.mean(crit)
#                     # Calculate the gradient
#                     adv_grad = torch.autograd.grad( loss,x_PN)[0]
#                     # Normalize the gradient values.
#                     adv_grad = normalize_grad(adv_grad, p=2, dim=1, eps=1e-12)
#                     x_PN = x_PN+ epsilon*adv_grad   
#                 # The critical cost function
#                 J_x_crit = criterion( model(x_PN.to(device), train_loader.edge_index.to(device))[mem_mask], train_loader.y[mem_mask].to(device) )
#                 # Derive gradients.
#                 # Update parameters based on gradients.
#                 ############### This is the loss J_th
#                 #########################################################################################
#                 opt_buffer = torch.optim.Adam(model.parameters(),lr = params['th_lr'])
#                 optimizer.zero_grad()
#                 with higher.innerloop_ctx(model, opt_buffer) as (fmodel, diffopt):
#                     for _ in range(params["theta_updates"]):
#                         loss_crit = criterion(fmodel(train_loader.x.to(device),\
#                                 train_loader.edge_index.to(device))[mem_mask],\
#                                 train_loader.y[mem_mask].to(device))
#                         loss_m = -1*torch.mean(loss_crit) 
#                         diffopt.step(loss_m)
#                     J_th_crit = (criterion(fmodel(train_loader.x.to(device),train_loader.edge_index.to(device))[mem_mask], train_loader.y[mem_mask].to(device)))
#                     Total_loss=torch.mean(J_M)+torch.mean(J_P) \
#                         + params['factor']*(torch.mean(J_x_crit) +torch.mean(J_th_crit))
#                     Total_loss.backward() 
#                 optimizer.step() 
#                 return Total_loss.detach().cpu(),\
#                     (torch.mean(J_M)+torch.mean(J_x_crit)).detach().cpu(),\
#                     (torch.mean(J_P)+torch.mean(J_th_crit)).detach().cpu()
#             else:
#                 out = model(train_loader.x.to(device), train_loader.edge_index.to(device))  # Perform a single fo
#                 ## Get loss on the memory and task and put it together
#                 critti = criterion(out[train_loader.train_mask], train_loader.y[train_loader.train_mask].to(device))
#                 Total_loss = torch.mean(critti)
#                 optimizer.zero_grad() 
#                 Total_loss.backward()  # Derive gradients.
#                 optimizer.step()  # Update parameters based on gradients.
#                 return Total_loss.detach().cpu(), Total_loss.detach().cpu(), Total_loss.detach().cpu()



def Node_update(model, criterion, optimizer, mem_loader, train_loader, task, \
            params = {'x_updates': 1,  'theta_updates':1, 'factor': 0.0001,\
                 'x_lr': 0.0001,'th_lr':0.0001,\
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
            'batchsize':8, 'total_updates': 1000} ):
        x, edge_, y, mask_t = train_loader
        for i in range(params['total_updates']): 
            if task>0:
                # print("1 did you copying throw an error")
                rand_index = random.randint(0,len(mem_loader)-1)
                # Send the data to the device
                data_m = mem_loader[rand_index].to(params['device'])
                out = model(x, edge_)  # Perform a single forward pass with the new data
                J_P = criterion(out[mask_t], y[mask_t])
                J_M = criterion(out[data_m], y[data_m])
                mem_mask=torch.logical_or(mask_t, data_m)
                import copy
                # print("4 did you copying throw an error")
                x_PN=copy.copy(x)
                x_PN.requires_grad = True
                epsilon = params['x_lr']
                # The x loop
                for epoch in range(params["x_updates"]):
                    x_PN = x_PN+epsilon*normalize_grad(torch.autograd.grad(\
                    torch.mean(criterion(\
                    model(x_PN, edge_)[mem_mask], y[mem_mask])),x_PN)[0],\
                    p=2, dim=1, eps=1e-12)
                # The critical cost function
                J_x_crit = criterion( model(x,edge_)[mem_mask], y[mem_mask])
                optimizer.zero_grad()
                # print("I entered the higher loop ")
                with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
                    for _ in range(params["theta_updates"]):
                        diffopt.step(-1*torch.mean(criterion(fmodel(x, edge_)[mem_mask],y[mem_mask])))
                    J_th_crit = (criterion(fmodel(x,edge_)[mem_mask], y[mem_mask]))
                    Total_loss=torch.mean(J_M)+torch.mean(J_P) \
                    + params['factor']*(torch.mean(J_x_crit)+torch.mean(J_th_crit))
                    Total_loss.backward() 
                # print("I exit the higher loop")
                optimizer.step() 
                return Total_loss.detach().cpu(),\
                    (torch.mean(J_M)+torch.mean(J_x_crit)).detach().cpu(),\
                    (torch.mean(J_P)+torch.mean(J_th_crit)).detach().cpu()
            else:
                # print("1here")
                # with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
                Total_loss = torch.mean(criterion(model(x, edge_)[mask_t], y[mask_t]))
                # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
                # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
                # print("2here")
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

def continuum_node_classification( datas, n_Tasks, num_classes, num_labels_task=1):
    dataset = datas[0]
    # print("features inside the start", dataset.x)
    n_labels = num_classes
    n_labels_per_task = num_labels_task
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
    # print(mask.sum())
    # print("1 In the accuracy calculateion")
    # print(x.shape, y.shape, edge.shape, mask.shape)
    model.eval()
    # print("2 In the accuracy calculateion")
    # print(x.shape, y.shape, edge.shape, mask.shape)
    out = model(x.to(d), edge.to(d)).cpu()
    # print("3 In the accuracy calculateion")
    # print(x.shape, y.shape, edge.shape, mask.shape)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[mask] == y[mask] # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
    # print("4 Out the accuracy calculateion")
    # print(x.shape, y.shape, edge.shape, mask.shape)
    return test_acc

def _f1_node(model, x, edge, y, mask, d='cuda'):
    #print(mask.sum())
    with torch.no_grad():
        # print("1 In the accuracy calculateion")
        out = model(x, edge).cpu()
        pred=torch.argmax(out,1)
        f1_ = f1_score(y[mask].detach().numpy(), pred[mask].detach().numpy(),average='micro' )
        #print("3 Out In the F1 calculation")
        return f1_


def test_NC(model, loader, masks, d="cuda"):
    # print("begin")
    # print(loader.x.shape, loader.y.shape, loader.edge_index.shape)
    x, edge_index, y, _ = loader
    if len(masks) ==1:
        # print(masks[0].shape)
        acc=[_Acc_node(model, x, edge_index, y.cpu(), masks[0].cpu())]
        # print("came out now")
        f1=[_f1_node(model, x, edge_index, y.cpu(), masks[0].cpu())]
    else:
        #print("In test NC")
        #print(len(masks))
        acc=[_Acc_node(model, x, edge_index, y.cpu(), masks[i].cpu()) for i in range(len(masks))]
        f1 =[_f1_node(model, x, edge_index, y.cpu(), masks[i].cpu()) for i in range(len(masks))]
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
    return train_loader, test_loader,\
    mem_train_loader, mem_test_loader, memory_train, memory_test 

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

    elif data_label=='cora' or data_label=='PubMed'\
        or data_label =='CiteSeer' or data_label=='cora_ML':
        from torch_geometric.datasets import CitationFull
        from torch_geometric.transforms import NormalizeFeatures
        dataset = CitationFull(root='data/CitationFull', name=data_label)
        data= dataset[0]
        print(data)
        print("from the load dataset", data.x)
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return dataset
    elif data_label=='Reddit':
        print(data_label)
        from torch_geometric.datasets import Reddit
        dataset = Reddit(root='data/Reddit')
        data= dataset[0]
        print(data)
        print("from the load dataset", data.x)
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return dataset

    elif data_label=='tox21':
        print(data_label)
        from torch_geometric.datasets import MoleculeNet
        dataset = MoleculeNet(root='data/tox21', name="tox21")
        data= dataset[0]
        print(data)
        print("from the load dataset", data.x)
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}')
        print(f'Number of classes: {dataset.num_classes}')
        return dataset



def runGraph(name_label, epochs, print_it, config, model,\
    criterion, optimizer, dataset):
    import random
    memory_train=[]
    memory_test=[]
    import torch.optim.lr_scheduler as lrs
    # scheduler = lrs.ExponentialLR(optimizer, gamma=0.9)
    accuracies_mem = []
    accuracies_one=[]
    F1_mem = []
    F1_one=[]

    Total_loss=[]
    Gen_loss=[]
    For_loss=[]
    n_Tasks=dataset.num_classes
    for i in range(n_Tasks):
        if config['full'] <1:
            # print("The task number", i)
            train_loader, test_loader, mem_train_loader, mem_test_loader, memory_train, memory_test = continuum_Graph_classification(dataset, memory_train, memory_test, batch_size=64, task_id=i)
            for epoch in range(1,(epochs*(i+1)) ):
                Total,Gen,For=train_CL(model, criterion, optimizer, mem_train_loader, train_loader, task=i, graph=1, node=0, params=config)
                Total_loss.append(Total)
                Gen_loss.append(Gen)
                For_loss.append(For)
                if epoch%print_it==0 and epoch>print_it:
                    # scheduler.step()
                    # train_acc, train_F1 = test_GC(model, train_loader)
                    test_acc, test_F1 = test_GC(model, test_loader)
                    # mem_train_acc, mem_train_f1 = test_GC(model, mem_train_loader)
                    mem_test_acc, mem_test_f1 = test_GC(model, mem_test_loader)
                    # print(test_F1, mem_test_f1)
                    print(f'Task: {i:03d}, Epoch: {epoch:03d}, Test Acc: {test_acc:.3f},  Mem Test Acc: {mem_test_acc:.3f}, Test F1: {test_F1:.3f}, Mem Test F1: {mem_test_f1:.3f}')
            # scheduler.step()
            # train_acc, train_F1 = test_GC(model, train_loader)
            test_acc, test_F1 = test_GC(model, test_loader)
            # mem_train_acc, mem_train_f1 = test_GC(model, mem_train_loader)
            mem_test_acc, mem_test_f1 = test_GC(model, mem_test_loader)
            # print(test_F1, mem_test_f1)
            accuracies_mem.append(mem_test_acc)
            accuracies_one.append(test_acc)
            F1_mem.append(mem_test_f1)
            F1_one.append(test_F1)
        else:
            dataset = dataset.shuffle()
            length = len(dataset)
            # print("what is length", length)
            train_dataset = dataset[:int(0.80*length)]
            test_dataset = dataset[int(0.80*length):]
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            for epoch in range(1, epochs):
                for data in train_loader:  # Iterate in batches over the training dataset.
                    out = model(data.x.to(config['device']), data.edge_index.to(config['device']), data.batch.to(config['device']))  # Perform a single forward pass.
                    loss = torch.mean(criterion(out, data.y.to(config['device']))) # Compute the loss.
                    loss.backward()  # Derive gradients.
                    optimizer.step()  # Update parameters based on gradients.
                    optimizer.zero_grad()  # Clear gradients.
                if epoch%print_it==0 and epoch>print_it:
                    # scheduler.step()
                    train_acc, train_F1 = test_GC(model, train_loader)
                    test_acc, test_F1 = test_GC(model, test_loader)
                    mem_train_acc, mem_train_f1 = test_GC(model, train_loader)
                    mem_test_acc, mem_test_f1 = test_GC(model, test_loader)
                    # print(test_F1, mem_test_f1)
                    print(f'Task: {i:03d}, Epoch: {epoch:03d}, Test Acc: {test_acc:.3f},  Mem Test Acc: {mem_test_acc:.3f}, Test F1: {test_F1:.3f}, Mem Test F1: {mem_test_f1:.3f}')
            
            #  scheduler.step()
            # train_acc, train_F1 = test_GC(model, train_loader)
            test_acc, test_F1 = test_GC(model, test_loader)
            # mem_train_acc, mem_train_f1 = test_GC(model, mem_train_loader)
            mem_test_acc, mem_test_f1 = test_GC(model, mem_test_loader)
            # print(test_F1, mem_test_f1)
            accuracies_mem.append(mem_test_acc)
            accuracies_one.append(test_acc)
            F1_mem.append(mem_test_f1)
            F1_one.append(test_F1)
            break


    PM, FM, AP, AF = metrics(accuracies_mem, F1_mem)
    # #After the task has been learnt
    if epoch>print_it:
        print("##########################################")
        print(f'PM: {PM:.3f}, FM: {FM:.3f}, AP: {AP:.3f}, AF: {AF:.3f}')
        print("##########################################")
    import numpy as np
    F1_one=np.array(F1_one).reshape([-1])
    F1_mem=np.array(F1_mem).reshape([-1])
    Total_loss= np.array(Total_loss).reshape([-1])
    Gen_loss= np.array(Gen_loss).reshape([-1])
    For_loss=np.array(For_loss).reshape([-1])

    accuracies_one = np.array(accuracies_one).reshape([-1])
    accuracies_mem=np.array(accuracies_mem).reshape([-1])
    print(accuracies_one.shape, accuracies_mem.shape, F1_one.shape, F1_mem.shape)
    del model, criterion, optimizer, memory_train, memory_test
    return accuracies_one, accuracies_mem,F1_one, F1_mem, PM, FM, AP, AF, Total_loss, Gen_loss, For_loss


def metrics(accuracy_mem, F1_mem):
    # The metrics from ER paper
    PM=round((sum(F1_mem)/len(F1_mem))*100,2)
    if len(accuracy_mem)>0:
        F1_shifted = F1_mem[:-1]
        F1_shifted.insert(0,F1_mem[0])
        diff = abs(np.subtract(F1_mem,F1_shifted))

        FM=round(max(diff)*100,2)
    else:
        FM=F1_mem[-1]

    # The metrics from ER paper
    AP=round( (sum(accuracy_mem)/len(accuracy_mem)*100),2)
    if len(accuracy_mem)>0:
        F1_shifted = accuracy_mem[:-1]
        F1_shifted.insert(0,accuracy_mem[0])
        diff = abs(np.subtract(accuracy_mem,F1_shifted))
        AF=round(max(diff)*100,2)
    else:
        AF=accuracy_mem[-1]
    return PM, FM, AP, AF 


      


def load_corafull(dataset):
    # load corafull dataset
    g = next(iter(dataset))
    label = g.y.numpy()
    label_counter = collections.Counter(label)
    selected_ids = [id for id, count in label_counter.items() if count > 150]
    np.random.shuffle(selected_ids)
    # print(g)
    print(f"selected {len(selected_ids)} ids from {max(label)+1}")
    mask_map = np.array([label == id for id in selected_ids])
    # set label to -1 and remap the selected id
    label = label * 0.0 - 1
    label = label.astype(np.int)
    for newid, remap_map in enumerate(mask_map):
        label[remap_map] = newid
        g.y = torch.LongTensor(label)
    mask_map = np.sum(mask_map, axis=0)
    mask_map = (mask_map >= 1).astype(np.int)
    mask_index = np.where(mask_map == 1)[0]
    np.random.shuffle(mask_index)
    train_mask = np.zeros_like(label)
    train_mask[ mask_index[ 0 : 40*len(selected_ids) ] ] = 1
    val_mask = np.zeros_like(label)
    val_mask[ mask_index[ 40*len(selected_ids) : 60*len(selected_ids) ] ] = 1
    test_mask = np.zeros_like(label)
    test_mask[ mask_index[ 60*len(selected_ids): ] ] = 1
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
    labels = g.y
    features = g.x
    return g.edge_index, features, labels, train_mask, val_mask, test_mask


def runNode(name_label, epochs, print_it, config, model,\
    criterion, optimizer, dataset):
    import random
    memory_train=[]
    memory_test=[]
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    accuracies_mem = []
    accuracies_one=[]
    F1_mem = []
    F1_one=[]
    # Total_loss=[]
    # Gen_loss=[]
    # For_loss=[]
    n_Tasks=config['n_Tasks']
    x = dataset[0].x
    y = dataset[0].y
    edge_index = dataset[0].edge_index 
    print(dataset[0].train_mask, dataset[0].test_mask)
    continuum_data = continuum_node_classification(dataset, n_Tasks,\
    num_classes=config['num_classes'], num_labels_task=config['num_labels_task'])
    # The arrays for data
    memory_train=[]
    memory_test=[]
    # import torch.optim.lr_scheduler as lrs
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,\
    # T_0=print_it, eta_min=1e-6)
    lrs = []
    from torch.profiler import profile, record_function, ProfilerActivity
    for id, task in enumerate(continuum_data):
        train_mask, _, test_mask = task
        # print(train_mask.sum(), test_mask.sum())

        memory_train.append(train_mask)
        memory_test.append(test_mask)
        epochs_T = epochs
        device = config['device']
        x=x.to(device)
        edge_index=edge_index.to(device)
        y=y.to(device)
        train_mask=train_mask.to(device)

        for epoch in range(epochs_T+1):
            # print(epoch)
            # print("I reached her e")
            train_loader= (x, edge_index, y, train_mask)
            if config['full'] <1:
                train_CL( model, criterion, optimizer,\
                memory_train, train_loader, task=id, \
                graph = 0, node=1, params = { 'x_updates': config['x_updates'],\
                'theta_updates': config['theta_updates'],\
                'factor': config['factor'], 'x_lr': config['x_lr'],\
                'th_lr':config['th_lr'],'device': device,\
                'batchsize':8, 'total_updates': config['total_updates']})
            else:
                out =model(x, edge_index)[train_mask]
                Total_loss = torch.mean(criterion(out, y[train_mask]))
                optimizer.zero_grad() 
                Total_loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.


        # scheduler.step()
        test_acc, test_F1 = test_NC(model, train_loader, [test_mask])
        mem_test_acc, mem_test_f1 = test_NC(model, train_loader, memory_test)
        accuracies_mem.append(mem_test_acc)
        accuracies_one.append(test_acc)
        F1_mem.append(mem_test_f1)
        F1_one.append(test_F1)

        if epoch>print_it:    
            print(f'Task: {id:03d}, Epoch: {epoch:03d}, Test Acc: {test_acc:.3f},\
            Mem Test Acc: {mem_test_acc:.3f}, Test F1: {test_F1:.3f},\
            Mem Test F1: {mem_test_f1:.3f}')

    PM, FM, AP, AF = metrics(accuracies_mem, F1_mem)
    #After the task has been learnt
    if epoch>print_it:
        print("##########################################")
        print(f'PM: {PM:.3f}, FM: {FM:.3f}, AP: {AP:.3f}, AF: {AF:.3f}')
        print("##########################################")
    F1_one=np.array(F1_one).reshape([-1])
    F1_mem=np.array(F1_mem).reshape([-1])
    accuracies_one = np.array(accuracies_one).reshape([-1])
    accuracies_mem=np.array(accuracies_mem).reshape([-1])
    print(accuracies_one.shape, accuracies_mem.shape, F1_one.shape, F1_mem.shape)
    del model, criterion, optimizer, memory_train, memory_test
    return accuracies_one, accuracies_mem,F1_one, F1_mem, PM, FM, AP, AF 





def Run_it(configuration: dict):
    args=configuration['model_parse']
    name_label=configuration['name_label']
    save_dir=configuration['save_dir']
    total_epoch=configuration['total_epoch']
    print_it=configuration['print_it']
    total_runs=configuration['total_runs']
    dataset= load_data(name_label)

    print(dataset)
    # 2print("data is", dataset[0], dataset)

    ## the following is my development
    if configuration['prob'] == 'graph_class':
        n_Tasks=dataset.num_classes
        args.num_classes = dataset.num_classes
        args.num_features = dataset.num_features
        params= {'x_updates':configuration['x_updates'],  'theta_updates': configuration['theta_updates'],
        'factor': configuration['factor'], 'x_lr': configuration['x_lr'],'th_lr':configuration['th_lr'],\
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
        'batchsize':configuration['batchsize'], 'full': configuration['full'], 'total_updates': configuration['total_updates']}
    elif configuration['prob'] == 'node_class':
        n_Tasks=dataset.num_classes//configuration['num_labels_task']
        params = {'x_updates': configuration['x_updates'], 'n_Tasks':n_Tasks,\
        'num_classes':dataset.num_classes,\
        'num_labels_task':configuration['num_labels_task'], 'theta_updates': configuration['theta_updates'],\
        'factor': configuration['factor'], 'x_lr': configuration['x_lr'],'th_lr':configuration['th_lr'],\
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
        'batchsize':configuration['batchsize'], 'full': configuration['full'],\
        'total_updates': configuration['total_updates']} 

    acc_one = np.zeros((total_runs,n_Tasks))
    acc_m = np.zeros((total_runs,n_Tasks))
    f1_one = np.zeros((total_runs,n_Tasks))
    f1_m = np.zeros((total_runs,n_Tasks))
    PM = np.zeros((total_runs,1))
    FM = np.zeros((total_runs,1))
    AP = np.zeros((total_runs,1))
    AF = np.zeros((total_runs,1))
    for i in range(total_runs):
        if configuration['model_tit'] == 'GCN':
            model = GCN_ours(hidden_channels=configuration['hidden_channels'],\
            num_node_features= dataset.num_features,\
            num_classes=dataset.num_classes,seed=i,\
            dropout=configuration['dropout'],\
            layers=configuration['layers']).to(params['device'])

        elif configuration['model_tit'] == 'GAT':
            model = GAT_ours(nfeat=dataset.num_node_features,\
                    nclass=dataset.num_classes,\
                    drop_rate=configuration['dropout'],\
                    hidden=configuration['hidden'],\
                    in_head=8,\
                    out_head=1).to(params['device'])
        
        elif configuration['model_tit'] == 'HGS':
            model = Model(args).to(params['device'])
            
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(),\
            lr=configuration['learning_Rate'], weight_decay=configuration['decay'])    
        
        ## the following is my development
        if configuration['prob'] == 'graph_class':
            acc_one[i,:], acc_m[i,:], f1_one[i,:], f1_m[i,:],\
            PM[i,0], FM[i,0], AP[i,0], AF[i,0], Total_loss, Gen_loss, For_loss  =runGraph(name_label, epochs=total_epoch,\
            print_it=print_it, config=params, model=model, criterion=criterion, optimizer=optimizer, dataset=dataset)

            if i ==0:
                T_arr=np.zeros((total_runs,Total_loss.shape[0]))
                G_arr=np.zeros((total_runs,Gen_loss.shape[0]))
                F_arr=np.zeros((total_runs,For_loss.shape[0]))
                T_arr[i,: ] = Total_loss
                G_arr[i,: ] = Gen_loss
                F_arr[i,: ] = For_loss

            else:
                T_arr[i,: ] = Total_loss
                G_arr[i,: ] = Gen_loss
                F_arr[i,: ] = For_loss

        elif configuration['prob'] == 'node_class':
            g, features, labels, train_mask, val_mask, test_mask = load_corafull(dataset)
            datas= [Data(x=features, y=labels,edge_index=g,\
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)]

            acc_one[i,:], acc_m[i,:], f1_one[i,:], f1_m[i,:],\
            PM[i,0], FM[i,0], AP[i,0], AF[i,0]  =runNode(name_label, epochs=total_epoch,\
            print_it=print_it, config=params, model=model, criterion=criterion, optimizer=optimizer, dataset=datas)


    if total_epoch>print_it:
        print("##################################################################################################")
        print(f'MEAN--PM: {np.mean(PM):.3f}, FM: {np.mean(FM):.3f}, AP: {np.mean(AP):.3f}, AF: {np.mean(AF):.3f}')
        print(f'STD--PM: {np.std(PM):.3f}, FM: {np.std(FM):.3f}, AP: {np.std(AP):.3f}, AF: {np.std(AF):.3f}')
        print("##################################################################################################")
        plot_save(acc_m, acc_one, save_dir, name_label, total_epoch, print_it, total_runs, n_Tasks)
        plot_save(f1_m, f1_one, save_dir, name_label+'f1', total_epoch, print_it, total_runs, n_Tasks)
        plot_save_loss(T_arr, G_arr, F_arr, save_dir, name_label+'loss')
    
    return (100-AF[i,0])


def provide_hps(filename, quantoo, n_params):
    import numpy as np 
    import pandas as pd
    from sdv.tabular import GaussianCopula
    df =pd.read_csv('results_fi/results.csv', delimiter=',', header=None)
    header=df.values[0,:]
    df =pd.read_csv(filename, delimiter=' ', header=None, names=header)
    q_10 = np.quantile(df.objective.values, quantoo)
    real_df = df.loc[df['objective'] > q_10].drop(columns=['elapsed_sec', 'duration', 'objective', 'id'])
    model = GaussianCopula()
    model.fit(real_df)
    new_data = model.sample(num_rows= n_params)
    return new_data.to_dict(orient='records')

    