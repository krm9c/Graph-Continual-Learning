
import torch
def f1_score(y_pred:torch.Tensor, y_true:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    print(y_true.ndim, y_pred.ndim)
    print(y_true, y_pred)
    assert y_pred.ndim == 2
    assert y_true.ndim == 1
    epsilon=1e-7
    y_true = F.one_hot(y_true, 1)
    y_pred = F.softmax(y_pred, dim=1)
    
    tp = (y_true * y_pred).sum(dim=0)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0)
    fp = ((1 - y_true) * y_pred).sum(dim=0)
    fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = ((2* (precision*recall) / (precision + recall + epsilon)))
    f1 = f1.clamp(min=0, max=1)
    # print("F1", f1 , "F1 mean", (1-f1.mean()))
    return (1-f1.mean())



def _f1_node(model, x, edge, y, mask, d='cuda'):
    # print("1 In the accuracy calculateion")
    out = model(x.to(d), edge.to(d))
    # print("2 In the F1 calculation")
    # f1_ = f1_score(out[mask].to(d),y[mask].to(d))
    from sklearn.metrics import f1_score  
    print(y[mask.shape], out[mask].shape)
    f1_=f1_score(y[mask].cpu().detach().numpy(),  out[mask])
    # print("3 Out In the F1 calculation")
    return f1_




import torch
import numpy as np


from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


import torch
from torch.nn import Linear
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


d = "cuda"
model = GCN(hidden_channels=16).to(d)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x.to(d), data.edge_index.to(d))  # Perform a single forward pass.
      loss = criterion(out[data.train_mask].to(d), data.y[data.train_mask].to(d))  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test():
      model.eval()
      out = model(data.x.to(d), data.edge_index.to(d))
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask].to(d)  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      from sklearn.metrics import f1_score  
      f1_=f1_score(data.y[data.test_mask].cpu().detach().numpy(),  pred[data.test_mask].cpu().detach().numpy(),average='micro')
      return test_acc, f1_


for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    print(test())