import sys 
sys.path.append('../GCL/')
from Lib import *
from model import *
import numpy as np
import torch


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
hps_dict = provide_hps('../scripts_graph/results_fi/filename_protein.csv', 0.95, n_params=20)
print(len(hps_dict))
acc_list=[ Run_it({'total_epoch': 500, 'print_it': 100, 'total_runs':1,\
                'hidden': 2,
                'decay':hps_dict[i]['decay'],\
                'learning_Rate':hps_dict[i]['lr'],\
                'hidden_channels':int(hps_dict[i]['hc']),'dropout':float(hps_dict[i]['dropout']),\
                'layers':int(hps_dict[i]['n_lays']),\
                'x_updates': int(hps_dict[i]['x_updates']),  'theta_updates': int(hps_dict[i]['th_updates']),\
                'factor': 1, 'x_lr': 1e-07,'th_lr':1e-07,\
                'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
                'batchsize':16, 'total_updates': int(hps_dict[i]['tot_updates']),\
                'name_label':'Reddit', 'save_dir':'Results/Reddit/',\
                'prob':'node_class','model_parse':args, 'full':0, 'model_tit': 'GAT', 'num_labels_task':1 }) for i in range(len(hps_dict)+1)]

# import matplotlib.pyplot as plt
# plt.hist(acc_list)
# np.savetxt('results_fi/acc_cora_DH.csv', np.array(acc_list), delimiter=',')
# plt.savefig('results_fi/fig_cora.png')

# # ## Cora
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
