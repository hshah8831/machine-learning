import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def mean_feature(X):
    #converting D * N to N * D
    X = X.T

    # computing mean feature wise resulting to dimension 1 * D
    x_bar = X.mean(0)

    # centering the input X
    X_cen = X - x_bar

    return X_cen.T

def pca(X,d):
    #computing mean
    X_cen = mean_feature(X.copy())

    #SVD of the  centered X
    U_x, s_x, V_x = np.linalg.svd(X_cen)

    #picking the d columns
    U = U_x[:, 0:d]
    #mean of teh subspace
    U_mean = np.mean(U, axis=1)
    #transformed popints
    Y = np.dot(U.T, X_cen)

    return U, U_mean, Y

# loading the matrices from the file
mat = scipy.io.loadmat('data.mat')
X = mat.get("X_Question1").copy()
U, U_mean, Y = pca(X, 2)
#print(Y)
plt.plot(Y[0], Y[1], 'ro')
plt.show()
