import numpy as np
import scipy.io as sio
import random as ran
import sys

def lagrangeMulandThresh(C, tol, max_passes, X, Y):
    alpha = np.zeros(shape=(X.shape[0],1)).astype('float')
    alpha_old = np.zeros(shape=(X.shape[0], 1)).astype('float')
    b = 0
    passes = 0

    i_sel = []
    j_sel = []

    while(passes <= max_passes):
        num_alpha_changed = 0
        for i in range(1,X.shape[0]):
            Ei = evaluate(X, Y, alpha, X[i], b) - Y[i,0]
            if(((Y[i,0] * Ei < -tol) and (alpha[i,0] < C)) or ((Y[i,0] * Ei > tol) and (alpha[i,0] > 0))):
                j = ran.randrange(1, X.shape[0])
                i_sel.append(i)
                j_sel.append(j)
                while(j == i):
                    j = ran.randrange(1, X.shape[0])
                Ej = evaluate(X, Y, alpha, X[j], b) - Y[j,0]

                alpha_old = alpha.copy()
                if(Y[i,0] != Y[j,0]):
                    L = max(0, alpha[j, 0] - alpha[i, 0])
                    H = min(C, C + alpha[j, 0] - alpha[i, 0])
                else :
                    L = max(0, alpha[j, 0] + alpha[i, 0] - C)
                    H = min(C, alpha[j, 0] + alpha[i, 0])
                if(L==H):
                    continue
                eta = 2 * np.inner(X[i], X[j]) - np.inner(X[i], X[i]) - np.inner(X[j], X[j])
                if(eta >= 0):
                    continue
                dummy = alpha[j, 0] - (Y[i, 0] * (Ej - Ei))/ eta
                alpha[j, 0] = dummy

                if(alpha[j, 0] > H):
                    alpha[j, 0] = H
                elif(alpha[j,0] < L):
                    alpha[j, 0] = L

                if(abs(alpha[j,0] - alpha_old[j,0]) < 0.00001):
                    continue

                alpha[i, 0] += Y[i, 0] * Y[j, 0] * (alpha_old[j, 0] - alpha[j, 0])
                b1 =  b - Ei - (Y[i, 0] * (alpha[i, 0] - alpha_old[i, 0]) * np.inner(X[i], X[i])) - (Y[j, 0] * (alpha[j, 0] - alpha_old[j, 0]) * np.inner(X[i], X[j]))
                b2 =  b - Ej - (Y[i, 0] * (alpha[i, 0] - alpha_old[i, 0]) * np.inner(X[i], X[j])) - (Y[j, 0] * (alpha[j, 0] - alpha_old[j, 0]) * np.inner(X[i], X[j]))
                if(0 < alpha[i, 0] and alpha[i, 0] < C):
                    b = b1
                elif(0 < alpha[j, 0] and alpha[j, 0] < C):
                    b = b2
                else:
                    b = (b1 + b2)/ 2
                num_alpha_changed += 1


        if(num_alpha_changed == 0):
            passes += 1
        else:
            passes = 0
    #print(i_sel)
    #print(j_sel)
    return alpha, b

def evaluate(X, Y, alpha, x, b):
    result = 0
    i = 0
    inner = np.inner(x, X)
    while i < X.shape[0]:
        result = result + (alpha[i,0] * float(Y[i,0]) * inner[i])
        i = i + 1
    result = result + b
    if(result >= 0):
        return 1
    return -1


'''
    makes one class 1 and rest -1
'''
def makeOnevsRest(Y, cl):
    for i in range(Y.shape[0]):
        if(Y[i, 0] == cl):
            Y[i, 0] = 1.0
        else:
            Y[i, 0] = -1.0
    return Y

'''
    sigma(alpha[i]*Y[i]*X[i])
'''
def getwieght(alpha, Y, X):
    i = 0
    result = [[0, 0, 0]]
    while i < X.shape[0]:
        result = result + (alpha[i, 0] * float(Y[i, 0]) * X[i])
        i = i + 1
    return result

def accuracy(Y_old, Y_new):
    hit = 0
    miss = 0
    for i in range(Y_old.shape[0]):
        if (Y_new[i, 0] == Y_old[i, 0]):
            hit += 1
        else:
            miss += 1
    return hit / (hit + miss)

def kCrossValidation(X, Y, k):
    acc_max = 0
    w0_max = None
    w1_max = None
    w2_max = None
    b0_max = 0
    b1_max = 0
    b2_max = 0

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
                    hold_in_X = X[(j*grps):end, :]
                    hold_in_Y = Y[(j*grps):end, :]
#                    print(hold_in_X.shape)
#                    print(hold_in_Y.shape)
                else:
                    hold_in_X = np.concatenate((hold_in_X, X[(j*grps):end]))
                    hold_in_Y = np.concatenate((hold_in_Y, Y[(j*grps):end]))
#        print(hold_in_X.shape)
#        print(hold_in_Y.shape)
#        print(np.asmatrix(hold_in_X).shape)
#        print(np.asmatrix(hold_in_Y).shape)
#        print(hold_out_X.shape)
#        print(hold_out_Y.shape)
        if(k == 2):
            acc, w0, w1, w2, b0, b1, b2 = findOptimal(hold_in_X, hold_in_Y, hold_out_X, hold_out_Y)
        else :
            acc, w0, w1, w2, b0, b1, b2 = findOptimal(hold_in_X, hold_in_Y, hold_out_X, hold_out_Y)
        if (acc > acc_max):
            acc_max = acc
            w0_max = w0
            w1_max = w1
            w2_max = w2
            b0_max = b0
            b1_max = b1
            b2_max = b2
        hold_out_X = None
        hold_out_Y = None
        hold_in_X = None
        hold_in_Y = None

    return w0_max, w1_max, w2_max, b0_max, b1_max, b2_max

def predict(w0, w1, w2, b0, b1, b2, X):
    Y = np.zeros(shape=(X.shape[0], 1)).astype('float')

    for i in range(X.shape[0]):
        val = np.inner(w0[0], X[i]) + b0
        if (val >= 0):
            Y[i, 0] = 0

    for i in range(X.shape[0]):
        val = np.inner(w1[0], X[i]) + b1
        if (val >= 0):
            Y[i, 0] = 1

    for i in range(X.shape[0]):
        val = np.inner(w2[0], X[i]) + b2
        if (val >= 0):
            Y[i, 0] = 2
    return Y

def findOptimal(X_trn, Y_trn, X_tst, Y_tst):
    C = [0.001, 0.01, 0.1, 1, 10]
    TOL = 1
    MAX_PASSES = 10
    alpha_min = None
    b = None
    acc_max = 0
    w0_max = None
    w1_max = None
    w2_max = None
    b0_max = None
    b1_max = None
    b2_max = None
    C_max = C[0]

    Y_trn_0 = makeOnevsRest(Y_trn.copy(), 0)
    Y_trn_1 = makeOnevsRest(Y_trn.copy(), 1)
    Y_trn_2 = makeOnevsRest(Y_trn.copy(), 2)

    for c in C:
        alpha0, b0 = lagrangeMulandThresh(c, TOL, MAX_PASSES, X_trn, Y_trn_0)
        alpha1, b1 = lagrangeMulandThresh(c, TOL, MAX_PASSES, X_trn, Y_trn_1)
        alpha2, b2 = lagrangeMulandThresh(c, TOL, MAX_PASSES, X_trn, Y_trn_2)

        w0 = getwieght(alpha0, Y_trn_0, X_trn)
        w1 = getwieght(alpha1, Y_trn_1, X_trn)
        w2 = getwieght(alpha2, Y_trn_2, X_trn)

        Y_tst_final = predict(w0,w1,w2,b0,b1,b2,X_tst)
        #print(alpha0.T)
        #print(alpha1.T)
        #print(alpha2.T)
        acc_tst = accuracy(Y_tst, Y_tst_final)

        print("C = ", c, ", Accuracy = ", acc_tst)
        print(Y_tst_final)
        if(acc_tst > acc_max):
            acc_max = acc_tst
            w0_max = w0
            w1_max = w1
            w2_max = w2
            b0_max = b0
            b1_max = b1
            b2_max = b2
            C_max = c
#    print(C_max)
#    print(predict(w0_max, w1_max, w2_max, b0_max, b1_max, b2_max,X_tst))
    return acc_max, w0_max, w1_max, w2_max, b0_max, b1_max, b2_max

mat = sio.loadmat("data.mat")

X_trn = mat.get("X_trn").copy()
Y_trn = mat.get("Y_trn").copy().astype(float)

X_tst = mat.get("X_tst").copy()
Y_tst = mat.get("Y_tst").copy()

# this is for simple learning
findOptimal(X_trn.copy(), Y_trn.copy(), X_tst.copy(), Y_tst.copy())

# this is for 5 cross validation learning
'''
w0, w1, w2, b0, b1, b2 = kCrossValidation(X_trn.copy(), Y_trn.copy(), 5)
print("Training Accuracy")
print(accuracy(Y_trn, predict(w0, w1, w2, b0, b1, b2, X_trn)))
print("Test Accuracy")
print(accuracy(Y_tst, predict(w0, w1, w2, b0, b1, b2, X_tst)))
'''