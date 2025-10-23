# =============================================================================
# Implementation of a GNN-based clustering model using DMoN loss.
# This file contains:
#   1. GCNEncoder: A two-layer GCN model with residual connections.
#   2. dmon_loss: The Deep Modularity Networks loss function.
#   3. gnn_clustering_with_dmon: The main training loop.
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import torch.nn.functional as F
import numpy as np


class GCNEncoder(torch.nn.Module):
    """
    A two-layer Graph Convolutional Network (GCN) encoder.

    This architecture includes a residual connection that adds the input features (x)
    and the outputs of both GCN layers (z1, z2). 

    Input: Graph data (x, edge_index).
    Output: Node embeddings (Z).
    """
    def __init__(self, data):
        super(GCNEncoder, self).__init__()
        k = data.k
        x = data.x

        self.x = nn.Parameter(x)
        
        self.conv1 = GCNConv(k, k)
        self.conv2 = GCNConv(k, k)

    def forward(self, edge_index):
        z1 = self.conv1(self.x, edge_index)

        z2 = torch.relu(z1)            
        z2 = self.conv2(z2, edge_index)  

        z = self.x + z1 + z2
        z = F.normalize(z, p=2, dim=1)

        return z


def dmon_loss(C, edge_index, num_nodes, num_clusters):
    """
    Calculates the DMoN (Deep Modularity Networks) loss.

    The loss is composed of two main parts:
    1. Modularity Loss: Encourages dense connections within clusters and sparse
       connections between them.
    2. Collapse Regularization: Prevents all nodes from being assigned to a
       single cluster.

    Args:
        C (torch.Tensor): Soft cluster assignment matrix of shape (num_nodes, num_clusters).
        edge_index (torch.Tensor): Graph edge index.
        num_nodes (int): Number of nodes.
        num_clusters (int): Number of clusters (k).

    Returns:
        torch.Tensor: The total DMoN loss value.
    """
    # Build a dense adjacency matrix A from the sparse edge_inde
    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), (num_nodes, num_nodes)).to_dense()

    d = A.sum(dim=1)  # Degree vector
    m = A.sum() / 2  # Total number of edges

    # Modularity loss component
    B = A - torch.outer(d, d) / (2 * m)
    modularity_loss = -torch.trace(C.T @ B @ C) / (2 * m)

    # Collapse regularization component
    cluster_sums = C.sum(dim=0)  
    collapse_reg = np.sqrt(num_clusters / num_nodes) * torch.norm(cluster_sums, p='fro') - 1

    total_loss = modularity_loss + collapse_reg
    return total_loss


def gnn_clustering_with_dmon(data, labels, num_epochs=10000, return_embdict=False): 
    """
    Main function to train the GCN model for clustering using DMoN loss.

    This function initializes the GCNEncoder, sets up an optimizer, and runs a
    training loop that includes an early stopping mechanism based on the ARI score.

    Args:
        data (torch_geometric.data.Data): The input graph data.
        labels (np.ndarray): True labels for calculating ARI (for evaluation).
        num_epochs (int): Maximum number of training epochs.
        return_embdict (bool, optional): If True, returns embeddings at specific epochs.

    Returns:
        tuple: (best_ari, final_embeddings, saved_embeddings_dict)
    """
    model = GCNEncoder(data)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    best_ari = -10  
    count = 0  # Early stopping steps

    epochs_to_save_embeddings = [0, 1000, 1800, 8180]
    saved_embeddings_dict = {} 

    for epoch in range(num_epochs):
        model.train()
        Z = model(data.edge_index)  
        C = F.softmax(Z, dim=1)  
        loss = dmon_loss(C, data.edge_index, Z.shape[0], data.k)

        if return_embdict and epoch in epochs_to_save_embeddings:
            saved_embeddings_dict[epoch] = Z

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch %20 == 0 and epoch != 0:
            model.eval() 
            # embeddings = Z.detach().cpu().numpy()
            # kmeans = KMeans(n_clusters=data.k, n_init=10, random_state=0).fit(embeddings)
            # yhat = kmeans.labels_
            
            yhat = torch.argmax(Z, dim=1)       
            ari_score = adjusted_rand_score(labels, yhat)
            # print(f"epoch {epoch}: {loss}, ARI:{ari_score}")  

            if ari_score == 1:
                best_ari = ari_score
                break

            if ari_score > best_ari:
                best_ari = ari_score
                count = 0
            elif epoch >= 1000:
                count += 1

        if count == 40: 
            break

    return best_ari, Z, saved_embeddings_dict
