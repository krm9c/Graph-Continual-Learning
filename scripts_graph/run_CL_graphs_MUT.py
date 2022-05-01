import sys 
sys.path.append('../GCL/')
from Lib import *
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
args = parser.parse_args()


## MUTAG
hps_dict = provide_hps('filename_.csv', 0.80, n_params=100)
acc_list=[ Run_it({'total_epoch': 3000, 'print_it': 100000000, 'total_runs':1,\
                'decay':hps_dict[i]['decay'],\
                'learning_Rate':hps_dict[i]['lr'],\
                'hidden_channels':int(hps_dict[i]['hc']),'dropout':float(hps_dict[i]['dropout']),\
                'layers':int(hps_dict[i]['n_lays']),\
                'x_updates': int(hps_dict[i]['x_updates']),  'theta_updates': int(hps_dict[i]['th_updates']),\
                'factor': 1, 'x_lr': 1e-06,'th_lr':1e-04,\
                'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
                'batchsize':16, 'total_updates': int(hps_dict[i]['tot_updates']),\
                'name_label':'MUTAG', 'save_dir':'../Results/mutag/',\
                'prob':'graph_class','model_parse':args, 'full':0, 'model_tit': 'GCN' }) for i in range(len(hps_dict))]
import matplotlib.pyplot as plt
plt.hist(acc_list)
np.savetxt('results_fi/acc_MUTAG_DH.csv', np.array(acc_list), delimiter=',')
plt.savefig('results_fi/fig_MUTAG.png')
