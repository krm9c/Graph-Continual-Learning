import ray
import json
import pandas as pd
from functools import partial
import torch
from torch.utils.data.dataset import random_split
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,  SAGEConv
from torch_geometric.nn import global_mean_pool

import sys 
sys.path.append('../')
sys.path.append('../../')
from Lib import *
from Node_Classification import *

class GAT(torch.nn.Module):
    def __init__(self, nfeat, in_head, out_head,\
        nclass, drop_rate=0.6, hidden=8, hidden_layers=3):
        super(GAT, self).__init__()
        self.n_layers=hidden_layers
        self.hid = hidden
        self.in_head = in_head
        self.out_head = out_head
        self.dropout=drop_rate
        self.conv1 = GATConv(nfeat, self.hid, heads=self.in_head, dropout=self.dropout)
        self.conv2 = GATConv(self.hid*self.in_head, nclass, concat=False,
                            heads=self.out_head, dropout=self.dropout)
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def Run_it(configuration: dict):
    import torch
    import numpy as np
    name_label=configuration['name_label']
    save_dir=configuration['save_dir']
    total_epoch=configuration['epoch']
    print_it=configuration['print_it']
    total_runs=configuration['total_runs']

    dataset= load_data(name_label)
    g, features, labels, train_mask, val_mask, test_mask = load_corafull(dataset)
    cora = [Data(x=features, y=labels,edge_index=g,\
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)]
    #print( cora)
    n_Tasks=dataset.num_classes//configuration['num_labels_task']
    acc_one = np.zeros((total_runs,n_Tasks))
    acc_m = np.zeros((total_runs,n_Tasks))
    f1_one = np.zeros((total_runs,n_Tasks))
    f1_m = np.zeros((total_runs,n_Tasks))
    print(acc_one.shape, acc_m.shape, f1_one.shape, f1_m.shape)
    for i in range(total_runs):
        # The data characteristics
        model = GAT(nfeat=dataset.num_node_features,\
                    nclass=dataset.num_classes,\
                    drop_rate=configuration['dropout'],\
                    hidden=configuration['hidden'],\
                    in_head=8,\
                    out_head=1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(), lr=configuration["learning_rate"],\
             weight_decay=configuration["decay"])


        ## the following is my development
        acc_one[i,:], acc_m[i,:], f1_one[i,:], f1_m[i,:] =run(name_label, epochs=total_epoch,\
        print_it=print_it, config={'x_updates': configuration['x_updates'], 'n_Tasks':n_Tasks,\
        'num_classes':dataset.num_classes,\
        'num_labels_task':configuration['num_labels_task'], 'theta_updates': configuration['theta_updates'],\
        'factor': configuration['factor'], 'x_lr': configuration['x_lr'],'th_lr':configuration['th_lr'],\
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
        'batchsize':configuration['batchsize'], 'total_updates': configuration['total_updates']} ,\
        model=model, criterion=criterion, optimizer=optimizer, dataset=cora)

        plot_save(acc_m, acc_one, save_dir, name_label, total_epoch, print_it, total_runs, n_Tasks)
        plot_save(f1_m, f1_one, save_dir, name_label+'f1', total_epoch, print_it, total_runs, n_Tasks)



if __name__ == "__main__":
    # We define a dictionnary for the default values
    Run_it({
        'hidden':8,\
        'decay':1e-10,\
        "dropout":0.6,\
        "learning_rate": 1e-03,\
        'x_updates': 100,\
        'theta_updates':100, \
        'factor': 1,\
        'x_lr': 1e-05,
        'th_lr': 1e-05,
        'batchsize':16,\
        'total_updates': 500,\
        'name_label':'CiteSeer',\
        'save_dir':'../CiteSeer/ITL',\
        'epoch':500,\
        'print_it':500,\
        'total_runs':100,
        'num_labels_task':2
    })