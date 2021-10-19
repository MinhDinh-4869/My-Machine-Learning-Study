from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T
print(X, original_label)
def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()
    
kmeans_display(X, original_label)


def EDist(a, b):
    return (a - b).dot(a - b)

def FindLabel(x, centers):
    MIN = EDist(x, centers[0])
    label = 0
    for i in range(1,len(centers)):
        temp = EDist(x, centers[i])
        if MIN > temp:
            MIN = temp
            label = i
    return label
def Kmean_init_centers(X,K):
    return X[np.random.choice(X.shape[0], K, replace=False)]

def Kmean_assign_label(X, centers):
    D = [FindLabel(X[i], centers) for i in range(len(X))]
    return np.array(D)

def UpdateCenters(X, label, K):
    centers = []
    for i in range(K):
        cluster = []
        for j in range(len(label)):
            if label[j] == i:
                cluster.append(X[j])
        centers.append(np.mean(np.array(cluster), axis = 0))
    return np.array(centers)

def HasConverge(old_center, new_center):
    return (set([tuple(a) for a in old_center]) == 
        set([tuple(a) for a in new_center]))

def Kmeans(X, K):
    centers = Kmean_init_centers(X, K)
    while True:
        label = Kmean_assign_label(X, centers)
        new_centers = UpdateCenters(X, label, K)
        if HasConverge(centers, new_centers):
            break
        centers = new_centers
    return centers, label
centers, label = Kmeans(X, K)
print(label)
print(centers)

kmeans_display(X, label)