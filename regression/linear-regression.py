import scipy.io
import numpy as np
import sys

def closedFormLR(X, Y):
    return np.dot(np.mat(np.dot(np.mat(np.dot(X.T, X)).I, X.T)), Y)

def closedFormRLR(X, Y, lam):
    #print(X.shape, Y.shape)
    return np.dot(np.mat(np.dot((np.mat(np.dot(X.T, X))+ (lam * np.identity(X.shape[1]))).I, X.T)), Y)

def predict(W, X):
    return np.dot(X, W)

def error(Y, Y_new):
    return np.mean(np.power(Y_new - Y,2))

def basisExp(X, n):
    temp = X
    X = np.insert(X, 0, 1, axis=1)
    for i in range(2,n + 1):
        X = np.hstack((X, np.power(temp, i)))
    return X

def runPolyLR(X_trn, Y_trn, X_tst, Y_tst):
    poly = [2,5,10,20]
    for i in poly:
        X = basisExp(X_trn, i)
        W = closedFormLR(X, Y_trn)
        print("polynomial to {} ".format(i))
        print("--------------------- weights---------------------------")
        print(W)

        Y_hat = predict(W, X)
        print("--------------------- training error--------------------")
        print(error(Y_trn, Y_hat))

        ###test
        X = basisExp(X_tst, i)
        Y_hat = predict(W, X)
        print("--------------------- test error------------------------")
        print(error(Y_tst, Y_hat))


def runPolyRLR(X_trn, Y_trn, X_tst, Y_tst, lam, poly):
    X = basisExp(X_trn, poly)
    W = closedFormRLR(X, Y_trn, lam)
    #print("--------------------- weights--------------------")
    #print(W)

    Y_hat_trn = predict(W, X)
    #print("--------------------- training error--------------------")
    #print(error(Y_trn, Y_hat_trn))

    ###test
    X = basisExp(X_tst, poly)
    Y_hat_tst = predict(W, X)
    #print("--------------------- test error--------------------")
    #print(error(Y_tst, Y_hat_tst))
    return (W,error(Y_trn, Y_hat_trn), error(Y_tst, Y_hat_tst))

def kCrossValidation(X, Y, k):
    min = sys.maxsize

    W_min_fold = None
    lam_min_fold = -100
    poly_min_fold = 2
    err_trn_min_fold = 0
    err_tst_min_fold = 0


    hold_out_X = None
    hold_out_Y = None
    hold_in_X = None
    hold_in_Y = None
    grps = X.shape[0]//k
    for i in range(0, k):
        for j in range(0, k):
            if(j == i):
                end = X.shape[0] if (j+1)*grps > X.shape[0] else (j+1)*grps
                hold_out_X = X[j*grps:end]
                hold_out_Y = Y[j*grps:end]
            else:
                end = X.shape[0] if (j + 1) * grps > X.shape[0] else (j + 1) * grps
                if(hold_in_X is None and hold_in_Y is None):
                    hold_in_X = X[(j*grps):end]
                    hold_in_Y = Y[(j*grps):end]
                else:
                    hold_in_X = np.append(hold_in_X, X[(j*grps):end])
                    hold_in_Y = np.append(hold_in_Y, Y[(j*grps):end])
        if(k == 2):
            W_min, lam_min, poly_min, err_trn_min, err_tst_min = findOptimal(np.asmatrix(hold_in_X),
                                                                             np.asmatrix(hold_in_Y),
                                                                             hold_out_X, hold_out_Y)
        else :
            W_min, lam_min, poly_min, err_trn_min, err_tst_min = findOptimal(np.asmatrix(hold_in_X).T,
                                                                             np.asmatrix(hold_in_Y).T,
                                                                             hold_out_X, hold_out_Y)
        if (err_tst_min < min):
            W_min_fold = W_min
            lam_min_fold = lam_min
            poly_min_fold = poly_min
            err_tst_min_fold = err_tst_min
            err_trn_min_fold = err_trn_min
            min = err_tst_min

        hold_out_X = None
        hold_out_Y = None
        hold_in_X = None
        hold_in_Y = None
    return W_min_fold, lam_min_fold, poly_min_fold, err_trn_min_fold, err_tst_min_fold



def findOptimal(X_trn, Y_trn, X_tst, Y_tst):
    min = sys.maxsize
    W_min = None
    lam_min = -100
    poly_min = 2
    err_trn_min = 0
    err_tst_min = 0
    poly = [2, 5, 10, 20]
    for i in range(-100, 201):
        for j in poly:
            W, err_trn, err_tst = runPolyRLR(X_trn, Y_trn, X_tst, Y_tst, i, j)
            if(err_tst < min):
                W_min = W
                lam_min = i
                poly_min = j
                err_tst_min = err_tst
                err_trn_min = err_trn
                min = err_tst

    return W_min, lam_min, poly_min, err_trn_min, err_tst_min



# loading the matrices from the file
mat = scipy.io.loadmat('linear_regression.mat')

print("---------------------------------------Part B-------------------------------------------------------------------")
runPolyLR(mat.get("X_trn").copy(), mat.get("Y_trn").copy(),mat.get("X_tst").copy(), mat.get("Y_tst").copy())


print("---------------------------------------Part C-------------------------------------------------------------------")
min = sys.maxsize
W_optimal = None
lam_optimal = 0
err_tst_optimal = 0
err_trn_optimal = 0
k_arr = [2,5,10,mat.get("X_trn").shape[0]]
for k in k_arr:
    W_min, lam_min, poly_min, err_trn_min, err_tst_min = kCrossValidation(mat.get("X_trn").copy(), mat.get("Y_trn").copy(), k)
    if(err_tst_min < min):
        min = err_tst_min
        W_optimal = W_min
        lam_optimal = lam_min
        err_trn_optimal = err_trn_min
        err_tst_optimal = err_tst_min

print("Optimal Weight : " , W_optimal)
print("Optimal Lambda : " , lam_optimal)
print("Optimal Train Error : " , err_trn_optimal)
print("Optimal Tst Error : " , error(mat.get("Y_tst").copy(), predict(W_optimal, basisExp(mat.get("X_tst").copy(), 10))))