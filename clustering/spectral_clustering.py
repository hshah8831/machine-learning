import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import math
from scipy.linalg import eigh
from sklearn.cluster import KMeans

def gaussian_kernel(x, y, sigma):
    #euclidean distance
    diff = np.linalg.norm(x-y, 2, 0)
    #final RBF
    return math.exp(- diff / sigma)

def create_A(X, sigma):
    (n, D) = X.shape
    #initializing n * n, matrix A with zeros
    A = np.zeros((n, n))
    #calulating affinity
    for i in range(n):
        for j in range(n):
            if(i != j):
                A[i,j] = gaussian_kernel(X[i], X[j], sigma)
    return A

def create_D(A):
    n = A.shape[0]
    D = np.zeros((n, n))
    #diagonal matrix
    for i in range(n):
        D[i,i] = np.sum(A[i])
    return D

def spec_clus(X, k, sigma):
    #affinityy matrix
    A = create_A(X,sigma)
    #diagonal matrix
    D = create_D(A)

    #Laplacian Matrix
    L = D - A

    #fetching the largest k eigen vectors
    w, v = eigh(L, eigvals=(0, k - 1))

    #normalizing the eigen vectors
    P_sqr = v ** 2

    row_sums = P_sqr.sum(axis=1)
    row_sums = np.sqrt(row_sums)

    if(np.all(row_sums != 0)):
        Q = v.T / row_sums
    else:
        Q = v.T
    Q = Q.T

    #performing the kmeans on the normalized eigenvectors
    kmeans =  KMeans(n_clusters=k).fit(Q)

    #plotting it against the original data
    plt.scatter(X[:,0],X[:,1], c=kmeans.labels_)
    plt.show()

def main():
    mat = scipy.io.loadmat('data.mat')
    X = mat.get("X_Question2_3").copy()
    for i in [0.001, 0.01, 0.1, 1]:
        spec_clus(X.T, 4, 1)

if __name__ == "__main__":
    main()