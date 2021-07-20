import numpy as np
import scipy.io
from SLSP import slsp
from construct_K2 import construct_K
from construct_L2 import construct_L


def main():
    # load data
    data_name = 'Lung'
    print data_name

    mat = scipy.io.loadmat(data_name + '.mat')
    X = mat['X']
    X = X.astype(float)
    y = mat['Y']
    y = y[:, 0]
    Parm = [1e-4, 1e-2, 1, 1e+2, 1e+4]

    p = len(X[0])
    n = len(X[:, 0])

    num_cluster = len(np.unique(y))

    K = construct_K(X)
    L = construct_L(X)
    count = 0
    idx = np.zeros((p, 125), dtype=np.int)
    for Parm1 in Parm:
        for Parm2 in Parm:
            for Parm3 in Parm:
                print count
                Weight = slsp(X, K=K, L=L, n_clusters=num_cluster, lambd=Parm1, alpha=Parm2, beta=Parm3)
                idx[0:p, count] = feature_ranking(Weight)
                count += 1


def feature_ranking(W):
    """
    This function ranks features according to the feature weights matrix W

    Input:
    -----
    W: {numpy array}, shape (n_features, n_classes)
        feature weights matrix

    Output:
    ------
    idx: {numpy array}, shape {n_features,}
        feature index ranked in descending order by feature importance
    """
    T = (W*W).sum(1)
    idx = np.argsort(T, 0)
    return idx[::-1]


if __name__ == '__main__':
    main()
