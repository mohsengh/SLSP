import numpy as np


def construct_K(X):
    n = len(X[:, 0])
    k = 7
    K = np.zeros([n, n])
    Dist = np.zeros([n, n])

    for i in range(0, n):
        for j in range(0, n):
            Dist[i, j] = np.linalg.norm(X[i, :] - X[j, :])

    idx = np.argsort(Dist, axis=1)

    for i in range(0, n):
        for j in range(0, n):
            sigma_i = Dist[i, idx[i, k]]
            sigma_j = Dist[j, idx[j, k]]
            K[i, j] = np.exp(-pow(Dist[i, j], 2) / (sigma_i * sigma_j))

    return K
