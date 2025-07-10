# =============================================================================
# Implementation of the Unsupervised Graph Encoder Embedding (GEE) algorithm.
# This file contains the UnSup_Gee class, which encapsulates the logic for
# unsupervised node clustering on graphs.
#
# The core logic is adapted from the original implementation found in the
# GraphEmd repository: https://github.com/cshen6/GraphEmd/tree/main
#
# This file contains:
#   1. UnSup_Gee: The main class that orchestrates the clustering process.
#   2. GEE: This is the supervised GEE algorithm.
#   3. GEE_unsup: The main unsupervised training loop that iteratively refines
#      clusters by alternating between embedding and clustering.
#   4. calculate_temp_score: A helper function to evaluate the quality of a
#      clustering result to select the best model from multiple runs.
# =============================================================================

import numpy as np
from numpy import linalg as LA
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from torch_geometric.utils import to_scipy_sparse_matrix

class UnSup_Gee(nn.Module):
  def __init__(self, E, n, k):
    """
    Initializes the GEE model.

    Args:
        E (array-like): Edge list of the graph, shape (num_edges, 3) -> [node_i, node_j, weight].
        n (int): Number of nodes in the graph.
        k (int): Number of clusters.
    """
    super(UnSup_Gee, self).__init__()
    self.X = E
    self.n = n
    self.K = k
    self.replicates = 4
    self.num_iter = 70

  def GEE(self, Y):
    """
    Computes graph embeddings based on given node labels (supervised step).
    This function aggregates neighborhood information weighted by class assignments.

    Args:
        Y (np.array): Node labels

    Returns:
        np.array: L2-normalized node embeddings, shape (n, k).
    """

    possibility_detected = False
    if Y.shape[1] > 1:
      k = Y.shape[1]
      possibility_detected = True
    else:

      k = Y[:,0].max() + 1


    nk = np.zeros((1,k))
    W = np.zeros((self.n,k))

    if possibility_detected:
      nk=np.sum(Y, axis=0)
      W=Y/nk
    else:
      for i in range(k):
        nk[0,i] = np.count_nonzero(Y[:,0]==i)

      for i in range(Y.shape[0]):
        k_i = Y[i,0]
        if k_i >=0:
          W[i,k_i] = 1/nk[0,k_i]


    Z = np.zeros((self.n,k))
    i = 0
    for row in self.X:
      [v_i, v_j, edg_i_j] = row
      v_i = int(v_i)
      v_j = int(v_j)
      if possibility_detected:
        for label_j in range(k):
          Z[v_i, label_j] = Z[v_i, label_j] + W[v_j, label_j]*edg_i_j
          if v_i != v_j:
            Z[v_j, label_j] = Z[v_j, label_j] + W[v_i, label_j]*edg_i_j
      else:
        label_i = Y[v_i][0]
        label_j = Y[v_j][0]

        if label_j >= 0:
          Z[v_i, label_j] = Z[v_i, label_j] + W[v_j, label_j]*edg_i_j
        if (label_i >= 0) and (v_i != v_j):
          Z[v_j, label_i] = Z[v_j, label_i] + W[v_i, label_i]*edg_i_j

    row_norm = LA.norm(Z, axis = 1)
    reshape_row_norm = np.reshape(row_norm, (self.n,1))
    reshape_row_norm[reshape_row_norm == 0] = 1e-10 
    Z = np.nan_to_num(Z/reshape_row_norm)

    return Z


  def GEE_unsup(self):
      """
      Performs the main unsupervised learning procedure for graph clustering.

      This method uses an Expectation-Maximization-like approach. It starts
      with random cluster initializations and iteratively alternates between
      generating embeddings (E-step, via GEE) and re-assigning clusters
      (M-step, via KMeans). This process is repeated for multiple replicates
      to find the best clustering result.

      Returns:
          tuple: A tuple containing:
              - Z (np.array): The final node embeddings from the best run.
              - Y (np.array): The final cluster assignments (labels) from the best run.
      """
      minSS=-1
      Z = None

      for i in range(self.replicates):
        Y_temp = np.random.randint(self.K, size=(self.n,1))
        for r in range(self.num_iter):
          Zt = self.GEE(Y_temp)
          kmeans = KMeans(n_clusters=self.K, max_iter = self.num_iter).fit(Zt)
          labels = kmeans.labels_ # shape(n,)
          # sum_in_cluster = kmeans.inertia_ # sum of distance within cluster (k,1)
          dis_to_centors = kmeans.transform(Zt)
          # adjusted_rand_score() needs the shape (n,)
          ari = adjusted_rand_score(Y_temp.reshape(-1,), labels)
          if ari == 1:
            break
          else:
            # we need labels to be the same shape as for Y(n,1) when assign
            Y_temp = labels.reshape(-1,1)
          # print(f"iteration {r}:  ARI = {ari}")

        # calculate score and compare with meanSS
        tmp = self.calculate_temp_score(dis_to_centors, labels)
        if (minSS == -1) or tmp < minSS:
          Z = Zt
          minSS = tmp
          Y = labels
      return  Z, Y

  def calculate_temp_score(self, dis_to_centors, labels):
    """
    Calculates a quality score for a given clustering result.

    The score is based on a weighted ratio of the normalized intra-cluster
    distance to the normalized inter-cluster distance. A lower score indicates
    a better, more compact, and well-separated clustering.

    Args:
        dis_to_centors (np.array): Matrix of distances from each node to
                                  every cluster centroid.
        labels (np.array): The cluster assignment for each node.

    Returns:
        float: The calculated clustering quality score.
    """
    label_count = np.bincount(labels)
    sum_in_cluster_squre = np.zeros((self.K,))

    dis_to_centors_squre = dis_to_centors**2

    for i in range(self.n):
      label = labels[i]
      sum_in_cluster_squre[label] += dis_to_centors_squre[i][label]

    # how to find out if the distance is squared, the current method doesn't do square root.
    sum_not_in_cluster = (np.sum(dis_to_centors_squre, axis=0) - sum_in_cluster_squre)**0.5

    sum_not_in_cluster_norm = sum_not_in_cluster/(self.n - label_count)
    sum_in_cluster_norm = sum_in_cluster_squre**0.5/label_count

    tmp = sum_in_cluster_norm / sum_not_in_cluster_norm * label_count / self.n
    tmp = np.mean(tmp) + 2*np.std(tmp)

    return tmp


