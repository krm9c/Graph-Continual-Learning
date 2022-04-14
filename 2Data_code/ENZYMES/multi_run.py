
import sys 
sys.path.append('..')


from Graph_classification import *

configuration = {'x_updates': 10,  'theta_updates':10, 'factor': 1e-05, 'x_lr': 1e-03,'th_lr':1e-03,\
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),\
            'batchsize':64, 'total_updates': 250} 


import numpy as np
total_epoch=1000
print_it=250
total_runs=1
n_Tasks=6
acc_one = np.zeros((total_runs,((total_epoch//print_it-1)*n_Tasks)))
acc_m = np.zeros((total_runs,((total_epoch//print_it-1)*n_Tasks)))
name_label='ENZYMES'
save_dir='../ENZYMES/'
for i in range(total_runs):
    acc_one[i,:], acc_m[i,:] =run(name_label, epochs=total_epoch, print_it=print_it, config=configuration)
plot_save(acc_m, acc_one, save_dir, name_label, total_epoch, print_it, total_runs, n_Tasks)