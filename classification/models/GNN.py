# =============================================================================
# Implementation of a GCN-based model for semi-supervised node classification.
# This file contains:
#   1. GCNEncoder: A two-layer GCN model with residual connections.
#   2. grad_norm: A utility to compute the total norm of model gradients.
#   3. train: The main training and evaluation loop, featuring early stopping
#      based on validation accuracy.
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from sklearn.metrics import adjusted_rand_score
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import torch, numpy as np



# class GCNEncoder(torch.nn.Module):
#     """
#     A two-layer Graph Convolutional Network (GCN) encoder.

#     This architecture includes a residual connection that adds the input features (x)
#     and the outputs of both GCN layers (z1, z2). 

#     Input: Graph data (x, edge_index).
#     Output: Node embeddings (Z).
#     """
#     def __init__(self, x, k):
#         super(GCNEncoder, self).__init__()
#         self.k = k
#         self.x = nn.Parameter(x)
        
#         self.conv1 = GCNConv(k, k)
#         self.conv2 = GCNConv(k, k)

#     def forward(self, edge_index):
#         z1 = self.conv1(self.x, edge_index)

#         z2 = torch.relu(z1)            
#         z2 = self.conv2(z2, edge_index)  

#         z = self.x + z1 + z2
#         z = F.normalize(z, p=2, dim=1)

#         return z

class GCNEncoder(torch.nn.Module):
    """
    A multi-layer Graph Convolutional Network (GCN) encoder.
    """
    def __init__(self, x, k, num_layers=3): 
        super(GCNEncoder, self).__init__()
        self.k = k
        self.x = nn.Parameter(x)
        self.num_layers = num_layers 

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(k, k))

    def forward(self, edge_index):
        layer_outputs = []
        
        current_z = self.x
        
        for i in range(self.num_layers):
            current_z = self.convs[i](current_z, edge_index)
            layer_outputs.append(current_z) 

            if i < self.num_layers - 1:
                current_z = torch.relu(current_z)

        z = self.x
        for out in layer_outputs:
            z = z + out
            
        z = F.normalize(z, p=2, dim=1)

        return z



def grad_norm(model, norm_type=2):
    """
    Computes the total L-p norm of a model's gradients.
    """
    total = 0.
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().data.norm(norm_type).item() ** norm_type
    return total ** (1. / norm_type)


def train(data, k=4, lr=0.01, num_epochs=10000, patience=50, concat = False, return_grad=False):
    """
    Trains and evaluates the GCNEncoder for semi-supervised node classification.


    Args:
        data: A graph data object containing features, labels, and masks.
        k (int): The number of classes.
        lr (float): The learning rate for the Adam optimizer.
        num_epochs (int): The maximum number of training epochs.
        patience (int): The number of epochs to wait for validation improvement
                        before early stopping.
        concat (bool): If True, concatenates features from an external model
                       (e.g., GEE) with this model's internal features.
        return_grad (bool): If True, returns loss and gradient norms from the
                            first 100 epochs for diagnostics.

    Returns:
        tuple: A tuple containing:
            - The trained model with the best validation weights.
            - The final logits produced by the best model.
            - Diagnostic information (loss, grad norms, val acc) if requested,
              otherwise None.
    """
    model = GCNEncoder(data.x, k, num_layers=3)

    print(f"--- Initializing GCN model with {model.num_layers} layers ---")
    # model = GCNEncoder(data.x, k)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state, wait = 0.0, None, 0
    loss_list, grad_list, val_list = [], [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.edge_index)                # (n,C)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])

        if return_grad and epoch<=100:
            loss_list.append(loss.item())

        loss.backward()

        if return_grad and epoch<=100:
            grad_list.append(grad_norm(model))
        
        optimizer.step()

        # --- evaluation ---
        model.eval()
        with torch.no_grad():
            pred = logits.argmax(dim=1) % k

            train_acc = accuracy_score(data.y[data.train_mask].cpu(),
                                       pred[data.train_mask].cpu())
            val_acc = accuracy_score(data.y[data.val_mask].cpu(),
                                     pred[data.val_mask].cpu())
        if return_grad and epoch<=100:
            val_list.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc, best_state, wait = val_acc, model.state_dict(), 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Loss {loss:.4f} "
                  f"| TrainAcc {train_acc:.3f} | ValAcc {val_acc:.3f}")

        if wait >= patience:
            print("Early stop triggered.")
            break

    # load the best model
    if best_state != None:
        model.load_state_dict(best_state)
    model.eval(); logits = model(data.edge_index)
    pred = logits.argmax(dim=1) % k
    test_acc = accuracy_score(data.y[data.test_mask].cpu(),
                              pred[data.test_mask].cpu())
    print(f"\nBest ValAcc {best_val_acc:.3f} | TestAcc {test_acc:.3f}")

    grad_inf = loss_list, grad_list, val_list if return_grad else None
    return model, logits, grad_inf 
