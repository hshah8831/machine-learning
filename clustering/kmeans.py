import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import random
import sys

TOLERENCE = 1e-5

def plot_graph(X, C, closest):
    #plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=closest)
    #plot the centroid of each class
    plt.scatter(C[:, 0], C[:, 1], c='r')
    plt.show()

def cost(X, C, closest):
    """returns the cost of the model"""
    dummy = [np.inner(X[i], C[closest[i]]) for i in range(len(closest))]
    return sum(dummy)

def closest_centroid(points, centroids):
    """returns an array containing the index of the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

def kmeans(X, k, r):
    (n, D) = X.shape
    loss_min = sys.maxsize
    C_min = None
    closest_min = None
    #repeat the model for given number of repetitions
    for rep in range(r):
        #get list of random integers in the given range
        random_ind = random.sample(range(n), k)
        #using random_ind fetch random data points
        C = X[random_ind, :]
        #contains the class number corresponding to the data point in that index
        closest = None
        #run until convergrence
        while(True):
            #class allocation
            closest = closest_centroid(X, C)
            #centroid updation
            C_new = move_centroids(X, closest, C)
            #check convergrence
            if(np.all((C_new - C) < TOLERENCE)):
                break
            C = C_new
        #calculate the cost of the current repetition
        loss = cost(X, C, closest)
        #check if less than global minimum cost
        if(loss < loss_min):
            C_min = C
            closest_min = closest
    #plot the graph using minimum centroids and allocation
    plot_graph(X, C_min, closest_min)
    return C_min, closest_min

def main():
    mat = scipy.io.loadmat('data.mat')
    X = mat.get("X_Question2_3").copy()
    kmeans(X.T, 4, 100)

if __name__ == "__main__":
    main()