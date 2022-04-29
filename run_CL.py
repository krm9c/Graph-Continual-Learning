import sys 
sys.path.append('GCL/')
from Lib import *
from model import *
import numpy as np
import torch
def Run_it(configuration: dict):
    name_label=configuration['name_label']
    save_dir=configuration['save_dir']

    total_epoch=configuration['total_epoch']
    print_it=configuration['print_it']
    total_runs=configuration['total_runs']


    dataset= load_data(name_label)

    print("data is", dataset[0], dataset)

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
            model = Model(args).to(args.device)
            


        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(model.parameters(),\
            lr=configuration['learning_Rate'], weight_decay=configuration['decay'])    
        
        ## the following is my development
        if configuration['prob'] == 'graph_class':

            acc_one[i,:], acc_m[i,:], f1_one[i,:], f1_m[i,:],\
            PM[i,0], FM[i,0], AP[i,0], AF[i,0]  =runGraph(name_label, epochs=total_epoch,\
            print_it=print_it, config=params, model=model, criterion=criterion, optimizer=optimizer, dataset=dataset)


        elif configuration['prob'] == 'node_class':
            g, features, labels, train_mask, val_mask, test_mask = load_corafull(dataset)
            datas= [Data(x=features, y=labels,edge_index=g,\
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)]

            acc_one[i,:], acc_m[i,:], f1_one[i,:], f1_m[i,:],\
            PM[i,0], FM[i,0], AP[i,0], AF[i,0]  =runNode(name_label, epochs=total_epoch,\
            print_it=print_it, config=params, model=model, criterion=criterion, optimizer=optimizer, dataset=datas)

    print("####################################################################################")
    print(f'MEAN--PM: {np.mean(PM):.3f}, FM: {np.mean(FM):.3f}, AP: {np.mean(AP):.3f}, AF: {np.mean(AF):.3f}')
    print(f'STD--PM: {np.std(PM):.3f}, FM: {np.std(FM):.3f}, AP: {np.std(AP):.3f}, AF: {np.std(AF):.3f}')
    print("####################################################################################")
    plot_save(acc_m, acc_one, save_dir, name_label, total_epoch, print_it, total_runs, n_Tasks)
    plot_save(f1_m, f1_one, save_dir, name_label+'f1', total_epoch, print_it, total_runs, n_Tasks)



import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='PROTEINS', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')

args = parser.parse_args()
print(args)


## MUTAG
# Run_it({'total_epoch': 30001, 'print_it': 2000, 'total_runs':5, 'decay':1e-10,'learning_Rate':1e-4,\
#         'hidden_channels':32, 'dropout':0.6, 'layers':5,\
#         'x_updates': 5,  'theta_updates':100,\
#         'factor': 1, 'x_lr': 0.0001,'th_lr':0.001,\
#         'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
#         'batchsize':16, 'total_updates': 500, 'name_label':'MUTAG', 'save_dir':'Results/mutag/',\
#         'prob':'graph_class', 'model_tit': 'GCN' })

# ## PROTEINS


Run_it({'total_epoch': 50000, 'print_it': 1000, 'total_runs':1, 'decay':1e-10,'learning_Rate':1e-4,\
        'hidden_channels':16, 'dropout':0.6, 'layers':10,\
        'x_updates': 5,  'theta_updates':5,\
        'factor': 1, 'x_lr': 1e-06,'th_lr':1e-04,\
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
        'batchsize':16, 'total_updates': 500, 'name_label':'PROTEINS', 'save_dir':'Results/proteins/',\
        'prob':'graph_class','model_parse':args, 'full':0, 'model_tit': 'GCN' })

## ENZYMES
# Run_it({'total_epoch': 5000, 'print_it': 1000, 'total_runs':5, 'decay':1e-10,'learning_Rate':1e-4,\
#         'hidden_channels':32, 'dropout':0.6, 'layers':5,\
#         'x_updates': 5,  'theta_updates':100,\
#         'factor': 1, 'x_lr': 0.0001,'th_lr':0.001,\
#         'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
#         'batchsize':16, 'total_updates': 500, 'name_label':'ENZYMES', 'save_dir':'Results/ENZYMES/',\
#         'prob':'graph_class', 'model_tit': 'GCN' })


# ## Cora
# Run_it({
#     'hidden':8,\
#     'decay':1e-10,\
#     "dropout":0.6,\
#     "learning_Rate": 1e-04,\
#     'x_updates': 1,\
#     'theta_updates': 100,\
#     'factor': 0.5,\
#     'x_lr': 1e-07,\
#     'th_lr': 1e-07,\
#     'batchsize':16,\
#     'total_updates': 500,\
#     'total_epoch':500,\
#     'name_label':'cora_ML', 'save_dir':'Results/cora/',\
#     'prob':'node_class', 'model_tit': 'GAT',
#     'print_it':5,\
#     'total_runs':5,\
#     'num_labels_task':2 })

# # Pubmed
# Run_it({
#     'hidden':8,\
#     'decay':1e-10,\
#     "dropout":0.6,\
#     "learning_Rate": 1e-04,\
#     'x_updates': 1,\
#     'theta_updates': 100,\
#     'factor': 0.5,\
#     'x_lr': 1e-07,\
#     'th_lr': 1e-07,\
#     'batchsize':16,\
#     'total_updates': 500,\
#     'total_epoch':500,\
#     'name_label':'PubMed', 'save_dir':'Results/PubMed/',\
#     'prob':'node_class', 'model_tit': 'GAT',
#     'print_it':5,\
#     'total_runs':5,\
#     'num_labels_task':2 })

# # CiteSeer
# Run_it({
#     'hidden':8,\
#     'decay':1e-10,\
#     "dropout":0.6,\
#     "learning_Rate": 1e-04,\
#     'x_updates': 1,\
#     'theta_updates': 100,\
#     'factor': 0.5,\
#     'x_lr': 1e-07,\
#     'th_lr': 1e-07,\
#     'batchsize':16,\
#     'total_updates': 500,\
#     'total_epoch':500,\
#     'name_label':'CiteSeer', 'save_dir':'Results/CiteSeer/',\
#     'prob':'node_class', 'model_tit': 'GAT',
#     'print_it':5,\
#     'total_runs':5,\
#     'num_labels_task':2 })

# # Reddit
# Run_it({
#     'hidden':8,\
#     'decay':1e-10,\
#     "dropout":0.6,\
#     "learning_Rate": 1e-04,\
#     'x_updates': 1,\
#     'theta_updates': 100,\
#     'factor': 0.5,\
#     'x_lr': 1e-07,\
#     'th_lr': 1e-07,\
#     'batchsize':16,\
#     'total_updates': 500,\
#     'total_epoch':500,\
#     'name_label':'Reddit', 'save_dir':'Results/Reddit/',\
#     'prob':'node_class', 'model_tit': 'GAT',
#     'print_it':5,\
#     'total_runs':5,\
#     'num_labels_task':2 })
