import numpy as np


def construct_L(X):
    n = len(X[:, 0])
    Dist = np.zeros([n, n])

    S_temp = np.zeros([n, n])

    for i in range(0, n):
        for j in range(0, n):
            Dist[i, j] = np.linalg.norm(X[i, :] - X[j, :])

    idx = np.argsort(Dist, axis=1)

    for i in range(0, n):
        for j in range(0, n):
            sigma_i = Dist[i, idx[i, 7]]
            sigma_j = Dist[j, idx[j, 7]]
            S_temp[i, j] = np.exp(-pow(Dist[i, j], 2) / (sigma_i * sigma_j))

    idx_new = idx[:, 0:6]

    S = np.zeros([n, n])
    for i in range(0, n):
        for j in range(1, 6):
            S[i, idx_new[i, j]] = S_temp[i, idx_new[i, j]]
            S[idx_new[i, j], i] = S_temp[i, idx_new[i, j]]

    D = np.zeros([n, n])
    D2 = np.zeros([n, n])
    for i in range(0, n):
        sum = 0
        for j in range(0, n):
            sum += S[i, j]
        D[i, i] = sum

        if sum > 1e-6:
            D2[i, i] = sum**(-0.5)
        else:
            D2[i, i] = sum

    L = D - S
    L = np.dot(D2, np.dot(L, D2))
    return L