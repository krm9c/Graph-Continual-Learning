import collections
from queue import Empty
import sys 
sys.path.append('../')
sys.path.append('../../')
from GCL.Lib import *


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
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
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

        # print("The sizes of things", (x.element_size()*x.nelement())/1024, (y.element_size()*y.nelement())/1024,\
        #     (edge_index.element_size()*edge_index.nelement())/1024, (train_mask.element_size()*train_mask.nelement())/1024 )

        for epoch in range(epochs_T+1):
            # print(epoch)
            # print("I reached her e")
            train_loader= (x, edge_index, y, train_mask)
            # print("2here")
            # out =model(x, edge_index)[train_mask]
            # print("3here")
            # Total_loss = torch.mean(criterion(out, y[train_mask]))
            # optimizer.zero_grad() 
            # Total_loss.backward()  # Derive gradients.
            # optimizer.step()  # Update parameters based on gradients.
            train_CL( model, criterion, optimizer,\
            memory_train, train_loader, task=id, \
            graph = 0, node=1, params = { 'x_updates': config['x_updates'],\
            'theta_updates': config['theta_updates'],\
            'factor': config['factor'], 'x_lr': config['x_lr'],\
            'th_lr':config['th_lr'],'device': device,\
            'batchsize':8, 'total_updates': config['total_updates']})

        scheduler.step()
        test_acc, test_F1 = test_NC(model, train_loader, [test_mask])
        # print("idhar aa gaya")
        mem_test_acc, mem_test_f1 = test_NC(model, train_loader, memory_test)
        print(f'Task: {id:03d}, Epoch: {epoch:03d}, Test Acc: {test_acc:.3f},\
            Mem Test Acc: {mem_test_acc:.3f}, Test F1: {test_F1:.3f},\
            Mem Test F1: {mem_test_f1:.3f}')


        accuracies_mem.append(mem_test_acc)
        accuracies_one.append(test_acc)
        F1_mem.append(mem_test_f1)
        F1_one.append(test_F1)


    # The metrics from ER paper
    PM=accuracies_one[-1]
    diff =[ abs(F1_mem[-1]-ele) for ele in F1_mem]
    # print(diff)
    # print(max(diff))
    FM=max(diff)
    # The metric from catastrophic Forgetting paper
    AP=accuracies_mem[-1]
    AF=abs(accuracies_mem[id]-accuracies_mem[id-1])
    #After the task has been learnt
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

