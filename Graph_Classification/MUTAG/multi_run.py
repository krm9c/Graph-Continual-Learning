import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv 
from torch_geometric.nn import global_mean_pool
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features,\
            num_classes, seed, dropout, layers):
        super(GCN, self).__init__()
        torch.manual_seed(seed)
        self.droprate = dropout
        self.layers=layers 
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        for i in range(self.layers):
            prev=x
            x = self.conv2(x, edge_index)
            x = x.relu()+prev
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=self.droprate, training=self.training)
        x = self.lin(x)
        return x

import sys 
sys.path.append('../')
sys.path.append('../../')
from Lib import *
from Lib import *
def Run_it(configuration: dict):
    import torch
    name_label=configuration['name_label']
    save_dir=configuration['save_dir']
    import numpy as np
    total_epoch=configuration['total_epoch']
    print_it=configuration['print_it']
    total_runs=configuration['total_runs']
    dataset= load_data(name_label)
    n_Tasks=dataset.num_classes
    acc_one = np.zeros((total_runs,(((total_epoch//print_it)-1)*n_Tasks)))
    acc_m = np.zeros((total_runs,(((total_epoch//print_it)-1)*n_Tasks)))
    f1_one = np.zeros((total_runs,(((total_epoch//print_it)-1)*n_Tasks)))
    f1_m = np.zeros((total_runs,(((total_epoch//print_it)-1)*n_Tasks)))
    params= {'x_updates':configuration['x_updates'],  'theta_updates': configuration['theta_updates'],
               'factor': configuration['factor'], 'x_lr': configuration['x_lr'],'th_lr':configuration['th_lr'],\
                    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
                    'batchsize':configuration['batchsize'], 'total_updates': configuration['total_updates']}
    for i in range(total_runs):
        
        model = GCN(hidden_channels=configuration['hidden_channels'],\
        num_node_features= dataset.num_features,\
        num_classes=dataset.num_classes,seed=i,\
        dropout=configuration['dropout'],\
        layers=configuration['layers']).to(params['device'])
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(),\
            lr=configuration['learning_Rate'], weight_decay=configuration['decay'])    
        
        ## the following is my development
        acc_one[i,:], acc_m[i,:], f1_one[i,:], f1_m[i,:] =run(name_label, epochs=total_epoch,\
        print_it=print_it, config=params,\
        model=model, criterion=criterion, optimizer=optimizer, dataset=dataset)

    plot_save(acc_m, acc_one, save_dir, name_label, total_epoch, print_it, total_runs, n_Tasks)
    plot_save(f1_m, f1_one, save_dir, name_label+'f1', total_epoch, print_it, total_runs, n_Tasks)




from Graph_classification import *
Run_it({'total_epoch': 10000, 'print_it':1000, 'total_runs':2, 'decay':1e-4,'learning_Rate':1e-4,\
        'hidden_channels':32, 'dropout':0.5, 'layers':3,\
        'x_updates': 10,  'theta_updates':10,\
        'factor': 1, 'x_lr': 0.0001,'th_lr':0.001,\
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
        'batchsize':16, 'total_updates': 5000, 'name_label':'MUTAG', 'save_dir':'../MUTAG/'})