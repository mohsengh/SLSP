import numpy as np
import math
import sklearn.cluster


def slsp(X, **kwargs):
    lambd = kwargs['lambd']
    alpha = kwargs['alpha']
    beta = kwargs['beta']
    K = kwargs['K']
    L = kwargs['L']
    n_clusters = kwargs['n_clusters']
    G = kmeans_initialization(X, n_clusters)
    n_samples, n_features = X.shape

    X_T = np.transpose(X)
    XX = np.dot(X_T, X)
    XLX = np.dot(X_T, np.dot(L, X))

    D = np.identity(n_features)
    I = np.identity(n_samples)

    maxIter = 100
    obj = np.zeros(maxIter)

    for iter_step in range(maxIter):
        M = np.linalg.inv(XX + alpha * XLX + beta * D + 1e-6*np.eye(n_features))
        H = I - np.dot(X, np.dot(M, X_T))
        GG = np.dot(G, np.transpose(G))
        G = updateG(K, G, GG, H, lambd)
        W = np.dot(M, np.dot(X_T, G))
        temp = np.sqrt((W*W).sum(1))
        temp[temp < 1e-16] = 1e-16
        temp = 0.5 / temp
        D = np.diag(temp)

        obj[iter_step] = np.linalg.norm(K - GG, 'fro')**2 + lambd * np.trace(np.dot(np.transpose(G), np.dot(H, G)))
        if iter_step >= 1 and math.fabs(obj[iter_step] - obj[iter_step - 1]) / math.fabs(obj[iter_step]) < 1e-3:
            break

        # print iter_step
    print iter_step
    return W


def updateG(K, G, GG, H, lambd):
    grad = np.asarray(np.dot(GG + (lambd / 2) * H - K, G))
    stepsize = 1
    G_n = G - stepsize * grad
    G_n[G_n < 0] = 0

    oldobj = np.linalg.norm(K - GG, 'fro') ** 2 + lambd * np.trace(np.dot(np.transpose(G), np.dot(H, G)))
    newobj = np.linalg.norm(K - np.dot(G_n, np.transpose(G_n)), 'fro') ** 2 + lambd * np.trace(np.dot(np.transpose(G_n), np.dot(H, G_n)))

    if newobj - oldobj > 0.1 * np.sum(grad * (G_n - G)):
        while True:
            stepsize *= 0.1

            G_n = G - stepsize * grad
            G_n[G_n < 0] = 0

            newobj = np.linalg.norm(K - np.dot(G_n, np.transpose(G_n)), 'fro') ** 2 + lambd * np.trace(np.dot(np.transpose(G_n), np.dot(H, G_n)))
            if newobj - oldobj <= 0.1 * 0.1 * np.sum(grad * (G_n - G)):
                G = G_n
                break
    else:
            G = G_n

    return G

def kmeans_initialization(X, n_clusters):
    """
    This function uses kmeans to initialize the pseudo label

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    n_clusters: {int}
        number of clusters

    Output
    ------
    Y: {numpy array}, shape (n_samples, n_clusters)
        pseudo label matrix
    """

    n_samples, n_features = X.shape
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300,
                                    tol=0.0001, precompute_distances=True, verbose=0,
                                    random_state=None, copy_x=True, n_jobs=1)
    kmeans.fit(X)
    labels = kmeans.labels_
    Y = np.zeros((n_samples, n_clusters))
    for row in range(0, n_samples):
        Y[row, labels[row]] = 1
    T = np.dot(Y.transpose(), Y)
    G = np.dot(Y, np.sqrt(np.linalg.inv(T)))
    G = G + 0.02*np.ones((n_samples, n_clusters))
    return G
