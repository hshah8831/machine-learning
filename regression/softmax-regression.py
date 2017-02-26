import math
import numpy as np
import scipy as s
import scipy.io as sio


def hotEncoding(Y):
    m = Y.shape[0]
    classes = len(np.unique(Y))
    Y_HE = np.zeros(shape=(m,classes)).astype('float')
    for i,v in enumerate(Y):
        for j in v:
            Y_HE[i,j] = 1
    return Y_HE

def basisExp(X, n):
    temp = X
    X = np.insert(X, 0, 1, axis=1)
    for i in range(2,n + 1):
        X = np.hstack((X, np.power(temp, i)))
    return X

def softmax(X, W):
    z = X.dot(W)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def prediction(X, W):
    probs = softmax(X,W)
    preds = np.argmax(probs,axis=1)
    return preds

def initialW(r,c):
    return np.zeros(shape=(r, c)).astype('float')

def converged(W_old, W_new, err):
    if(np.all((W_old - W_new) >= -err) and np.all((W_old - W_new) <= err)):
        return True
    else:
        return False

def gradient(X, Y, W):
    m = X.shape[0]
    prob = softmax(X, W)
    loss = (-1 / m) * np.sum(Y * np.log(prob))
    grad = (-1 / m) * np.dot(X.T, (Y - prob))
    return loss, grad

def iterations(X, Y):
    X = basisExp(X,1)
    Y = hotEncoding(Y)
    W_old = initialW(X.shape[1], 3)
    losses = []
    loss_old = 0.0
    lr = 1e-3
    err = 1e-6
    while True:
        loss_new, grad = gradient(X, Y, W_old)
        losses.append(loss_new)
        W_new = W_old - lr * grad
        if (math.fabs(loss_old - loss_new) < err):
            break
        W_old = W_new
        loss_old = loss_new
    return W_new


def accuracy(X, Y, W):
    X = basisExp(X, 1)
    pred = prediction(X, W)
    pred = pred.reshape((Y.shape[0], Y.shape[1]))
    accuracy = sum(pred == Y) / (float(len(Y)))
    return accuracy


mat = sio.loadmat('logistic_regression.mat')

W = iterations(mat.get("X_trn").copy(), mat.get("Y_trn").copy())
print("---------------------------------Weight Matrix[[1,x1,x2][class1, class2, class3]]-------------------------------")
print(W)

acc_trn = accuracy(mat.get("X_trn").copy(), mat.get("Y_trn").copy(), W)
acc_tst = accuracy(mat.get("X_tst").copy(), mat.get("Y_tst").copy(), W)
print("---------------------------------Training accuracy--------------------------------------------------------------")
print(acc_trn * 100)
print("---------------------------------Testing accuracy---------------------------------------------------------------")
print(acc_tst * 100)
print("---------------------------------Training error percentage------------------------------------------------------")
print(100 * (1 - acc_trn))
print("---------------------------------Testing error percentage-------------------------------------------------------")
print(100 * (1 - acc_tst))