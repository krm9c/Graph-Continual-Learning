import sys 
sys.path.append('..')
from Lib import *
from Graph_classification import *


def Run_it(configuration: dict):
    import torch
    from torch.nn import Linear
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, ChebConv, SAGEConv
    from torch_geometric.nn import global_mean_pool
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
                x = self.conv2(x, edge_index)
                x = x.relu()
            # 2. Readout layer
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
            # 3. Apply a final classifier
            x = F.dropout(x, p=self.droprate, training=self.training)
            x = self.lin(x)
    #         return x
    

    import numpy as np
    total_epoch=10000
    print_it=2500
    total_runs=2
    n_Tasks=2
    acc_one = np.zeros((total_runs,((total_epoch//print_it-1)*n_Tasks)))
    acc_m = np.zeros((total_runs,((total_epoch//print_it-1)*n_Tasks)))
    name_label='MUTAG'
    save_dir='../MUTAG/'
    for i in range(total_runs):
        dataset= load_data(name_label)

        model = GCN(hidden_channels=configuration['hidden_channels'],\
        num_node_features= dataset.num_features,\
        num_classes=dataset.num_classes,seed=i,\
            dropout=configuration['dropout'], layers=configuration['layers']).to(configuration['params']['device'])
        

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(),\
            lr=configuration['learning_Rate'], weight_decay=configuration['decay'])    
        
        ## the following is my development
        acc_one[i,:], acc_m[i,:] =run(name_label, epochs=total_epoch,\
        print_it=print_it, config=configuration['params'],\
        model=model, criterion=criterion, optimizer=optimizer, dataset=dataset)
    plot_save(acc_m, acc_one, save_dir, name_label, total_epoch, print_it, total_runs, n_Tasks)

Run_it({'decay':1e-3,'learning_Rate':1e-4,\
                    'hidden_channels':32, 'dropout':0.5, 'layers':2,
                   'params': {'x_updates': 1,  'theta_updates':1, 'factor': 1, 'x_lr': 0.0001,'th_lr':0.0001,\
                    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
                    'batchsize':32, 'total_updates': 1000} })