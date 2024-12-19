import numpy as np
import torch
import torch.nn.functional as F


def weighted_kmean(w, weights, k, max_iter = 100, tol = 1e-4, init = 'random', random_state = None):
    """
    Performs weighted K means given a matrix and return the centriods. Random initialization is used

    args:
        w: torch.Tensor (n_samples, n_features)
        weights: torch.Tensor (n_samples,)
        k: int
    output:
        centriods: torch.Tensor (k, n_features)

    psuedo code:
    1. Choose K samples as initial centriods randomly
    2. calculate the euclidean distance between each sample and the centriods
    3. Assign a point to the closes centriod
    4. update the centriod of each cluster as weighted some of the points in the cluster
    5. Repeat steps 2-4 until convergence or max iterations are reached
    """

    n_samples, n_features = w.shape

    if init == 'random':
        np.random.seed(random_state)
        centriods = w[np.random.choice(n_samples, k, replace = False)]
    for i in range(max_iter):
        distances = torch.cdist(w, centriods)
        labels = torch.argmin(distances, dim = 1)
        new_centroids = torch.zeros_like(centriods)
        for j in range(k):
            mask = labels == j
            new_centroids[j] = torch.sum(w[mask] * weights[mask].view(-1, 1), dim = 0) / torch.sum(weights[mask])
        
        center_shift = (new_centroids - centriods).pow(2).sum()
        centriods = new_centroids
        if center_shift <= tol:
            break

    return centriods


def get_centriods(w, h, v, k):
    """
    Takes in the weight matrix, hessian matrix, feature length and number of centriods and returns the centriods

    args:
        w: torch.Tensor (M, N)
        h: torch.Tensor (N, N)
        v: int : Choose v such that M % v == 0 for now
        k: int
    output:
        centriods: torch.Tensor (k, N)
    

    Psuedocode:
        1. Rearrange the weight as (N x M/v, v)
        2. get the diagonal hessian and each value is duplicated v times, so that shape is (N x M/v, 1)
        3. call the function weighted_kmean with 1, 2 and k as arguments
        4. return the centriods
    
    w = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    w' = [[1, 4], [7, 8], [2, 5], [8, 11], [3, 6], [9, 12]]

    w.t() = [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]]
    w.t().view(3, -1, 2) = [[[1, 4], [7, 10]], [[2, 5], [8, 11]], [[3, 6], [9, 12]]]

    h = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    h.diag() = [1, 5, 9]
    h.diag().repeat(2, 1).t().contiguous().view(-1, 1) = [[1], [1], [5], [5], [9], [9]]

    """
    n_samples, n_features = w.shape
    n_rows = n_features * (n_samples // v)
    w = w.permute(1, 0).reshape(n_rows, v)
    h = h.diag().repeat(n_samples//v, 1).t().contiguous().view(-1, 1)
    return weighted_kmean(w, h, k)