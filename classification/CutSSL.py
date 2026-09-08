import numpy as np
from scipy import sparse

import graphlearning as gl
from graphlearning import ssl, utils, graph

from tqdm import tqdm

def conjgradfn(A, b, x0=None, max_iter=100, tol=1e-10):
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0

    r = b - A(x)
    p = r
    rsold = np.sum(r**2,axis=0)

    err = 1
    i = 0
    while (err > tol) and (i < max_iter):
        i += 1
        Ap = A(p)
        alpha = rsold / np.sum(p*Ap,axis=0)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.sum(r**2,axis=0)
        err = np.sqrt(np.sum(rsnew))
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x

class cut_ssl(ssl.ssl):
    def __init__(self, W=None, class_priors=None, s=[0.0,0.1], maxiter=100):
        """CutSSL
        ===================

        Semi-supervised learning via combinatorial solutions to a cardinality-constrained min-cut problem

        Parameters
        ----------
        W : numpy array, scipy sparse matrix, or graphlearning graph object (optional), default=None
            Weight matrix representing the graph.
        class_priors : numpy array (optional), default=None
            Class priors (fraction of data belonging to each class). If provided, the predict function
            will attempt to automatic balance the label predictions to predict the correct number of
            nodes in each class.
        s : iterable (optional), default=[0.0,0.1]
            Sequence of coefficients for diagonally perturbing L.

        References
        ---------
        """
        super().__init__(W, class_priors)

        self.s = s
        self.maxiter = maxiter
        #Setup accuracy filename
        fname = '_cutssl'
        self.name = 'CutSSL'

        self.accuracy_filename = fname

    def mu1(self, b, y, lamb, mu2, c, o, m, I2Aio):
      on, ok= o
      mu = 1/c*((y + b - lamb - mu2@ok.T).T@I2Aio - m)
      return mu

    def mu2(self, b, y, lamb, mu1, o, Aon, k):
      on, ok= o
      return 1/k*((y + b - lamb -on@mu1.T)@ok - Aon)

    def yklambk(self, xk, lambk):
        yk = np.maximum((lambk + xk), 0.)
        lambk = lambk + (xk - yk)
        return yk, lambk

    def Ainv(self, A, b):
      b = np.array(b)
      #Preconditioner
      m = A.shape[0]
      M = A.diagonal()
      M = sparse.spdiags(1/np.sqrt(M+1e-10),0,m,m).tocsr()

      #Conjugate gradient solver
      v = gl.utils.conjgrad(M*A*M, M*b, tol=10e-5, max_iter=200)
      xs = M*v
      return xs

    def _fit(self, train_ind, train_labels, all_labels=None):
        G = self.graph
        D = G.degree_matrix()

        #Get some attributes
        n = G.num_nodes
        M = n - len(train_labels)
        unique_labels = np.unique(train_labels)

        k = len(unique_labels)

        #tau + Graph Laplacian and one-hot labels
        L = G.laplacian(normalization='combinatorial')
        F = utils.labels_to_onehot(train_labels, k)

        #Locations of unlabeled points
        idx = np.full((n,), True, dtype=bool)
        idx[train_ind] = False

        #Left hand side matrix
        A0 = L[idx,:]
        A0 = A0[:,idx]

        #Right hand side
        b = -L[:,train_ind]*F
        b = b[idx,:]

        on = np.ones((M,1))
        ok = np.ones((k,1))
        o = (on, ok)
        m = n*self.class_priors - F.sum(0)
        m = np.expand_dims(m, 1)

        _xp = 1/M * np.ones((M,1))@m.T
        
        #Conjugate gradient solver
        _P = lambda x : x - 1/n * on @ (on.T@x)
        _A = lambda x : _P(A0@(_P(x)))
        _xk = conjgradfn(_A, _P(b), max_iter=100, tol=1e-6)
        _xk = _xk + _xp

        lambk = np.zeros_like(_xk)
        yk = np.maximum(lambk + _xk, 0)

        for j, _s in enumerate(self.s):
          A = L - _s*D
          b = -A[:,train_ind]*F
          b = b[idx,:]
          A = A[idx][:,idx]
          I2A = A + sparse.eye(M)

          I2Aio = self.Ainv(I2A, on)
          Aon = I2A@on
          c = on.T@I2Aio

          mu2k = np.zeros((M,1))
          lambk = np.zeros_like(_xk)
          yk = np.maximum(lambk + _xk, 0)
          for i in tqdm(range(self.maxiter[j])):
            mu1k = self.mu1(b, yk, lambk, mu2k, c, o, m, I2Aio)
            mu2k = self.mu2(b, yk, lambk, mu1k, o, Aon, k=k)

            _xk = self.Ainv(I2A, yk + b - on@mu1k.T - mu2k@ok.T - lambk)
            yk = np.maximum(lambk + _xk, 0)
            lambk = lambk + (_xk - yk)

            #Add labels back into array
            u = np.zeros((n,k))
            u[idx,:] = yk
            u[train_ind,:] = F

            #Compute accuracy if all labels are provided
            if (all_labels is not None) and (i%25 == 0):
                acc = gl.ssl.ssl_accuracy(u.argmax(1),true_labels=all_labels,train_ind=train_ind)
                print('%d, Accuracy = %.2f'%(i,acc))

        return u
