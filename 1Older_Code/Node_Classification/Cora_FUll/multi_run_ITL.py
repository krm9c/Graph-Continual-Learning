from errno import EDEADLK
import sys 
import traceback
sys.path.append('../')
sys.path.append('../../')
from Lib import *
from Node_Classification import *

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATv2Conv as GATConv
import time
class GAT(torch.nn.Module):
    def __init__(self, nfeat, in_head, out_head,\
        nclass, drop_rate=0.6, hidden=8, hidden_layers=3):

        super(GAT, self).__init__()
        self.hid = hidden
        self.in_head = in_head
        self.out_head = out_head
        self.dropout=drop_rate
        self.conv1 = GATConv(nfeat, self.hid, heads=self.in_head, dropout=self.dropout)
        self.conv2 = GATConv(self.hid*self.in_head, self.hid, heads=self.in_head, dropout=self.dropout)
        self.conv3 = GATConv(self.hid*self.in_head, nclass, concat=False,heads=self.out_head, dropout=self.dropout)
    def forward(self, x, edge_index):
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x


def Run_it(configuration: dict):
    import gc
    name_label=configuration['name_label']
    save_dir=configuration['save_dir']
    total_epoch=configuration['epoch']
    print_it=configuration['print_it']
    total_runs=configuration['total_runs']
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    PM = np.zeros((total_runs,1))
    FM = np.zeros((total_runs,1))
    AP = np.zeros((total_runs,1))
    AF = np.zeros((total_runs,1))
    
    #print(acc_one.shape, acc_m.shape, f1_one.shape, f1_m.shape)
    for i in range(total_runs):
        # The data characteristics
        model = GAT(nfeat=dataset.num_node_features,\
                    nclass=dataset.num_classes,\
                    drop_rate=configuration['dropout'],\
                    hidden=configuration['hidden'],\
                    in_head=8,\
                    out_head=1).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=configuration["learning_rate"],\
             weight_decay=configuration["decay"])
        print(torch.cuda. list_gpu_processes())

        x = cora[0].x.to(device)
        y=cora[0].y.to(device)
        edge_index=cora[0].edge_index.to(device)
        train_mask=cora[0].train_mask.to(device)
        val_mask=cora[0].val_mask.to(device)
        test_mask=cora[0].test_mask.to(device)

        def train():
            model.train()
            optimizer.zero_grad()  # Clear gradients.
            out = model(x, edge_index)  # Perform a single forward pass.
            loss = criterion(out[train_mask], y[train_mask])  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            return loss

        def test(mask):
            model.eval()
            out = model(x, edge_index)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct = pred[mask] == y[mask]  # Check against ground-truth labels.
            acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
            return acc


        for epoch in range(1, 10001):
            loss = train()
            val_acc = test(val_mask)
            test_acc = test(test_mask)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

        ## the following is my development
        # try:
        #     acc_one[i,:], acc_m[i,:], f1_one[i,:], f1_m[i,:] = run(name_label, epochs=total_epoch,\
        #     print_it=print_it, config={'x_updates': configuration['x_updates'], 'n_Tasks':n_Tasks,\
        #     'num_classes':dataset.num_classes,\
        #     'num_labels_task':configuration['num_labels_task'], 'theta_updates': configuration['theta_updates'],\
        #     'factor': configuration['factor'], 'x_lr': configuration['x_lr'],'th_lr':configuration['th_lr'],\
        #     'device': device,\
        #     'batchsize':configuration['batchsize'], 'total_updates': configuration['total_updates']} ,\
        #     model=model, criterion=criterion, optimizer=optimizer, dataset=cora)
        # except RuntimeError as e:
        #     print("The error message", e)
        #     print(traceback.format_exc())
            #print(torch.cuda.mem_get_info())
            #print(torch.cuda.memory_summary())
            #print(torch.cuda.memory_snapshot())

        # print( '(',np.mean(acc_one[:,-1], axis=1),',', np.std(acc_one[:,-1], axis=1), ')')
        # print( '(',np.mean(acc_m[:,-1], axis=1),',', np.std(acc_m[:,-1], axis=1), ')')
        # print( '(',np.mean(f1_m[:,-1], axis=1),',', np.std(f1_m[:,-1], axis=1), ')')
        # print( '(',np.mean(f1_one[:,-1], axis=1),',', np.std(f1_one[:,-1], axis=1), ')')

        del model
        model=None
        criterion=None
        optimizer=None
        gc.collect()
        time.sleep(3)
        torch.cuda.empty_cache()
        plot_save(acc_m, acc_one, save_dir, name_label, total_epoch, print_it, total_runs, n_Tasks)
        plot_save(f1_m, f1_one, save_dir, name_label+'f1', total_epoch, print_it, total_runs, n_Tasks)

if __name__ == "__main__":
    # We define a dictionnary for the default values
    Run_it({
        'hidden':8,\
        'decay':5e-04,\
        "dropout":0.4,\
        "learning_rate":  0.005,\
        'x_updates': 0,\
        'theta_updates': 0,\
        'factor': 0,\
        'x_lr': 1e-30,\
        'th_lr': 1e-200,\
        'batchsize':16,\
        'total_updates': 5000,\
        'epoch':100,\
        'name_label':'cora',\
        'save_dir':'../Cora/ITL',\
        'print_it':5,\
        'total_runs':1,\
        'num_labels_task':5 })