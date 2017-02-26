import scipy.io as sio
import numpy as np
import math

mat = sio.loadmat("data.mat")

X_trn = mat.get("X_trn").copy()
Y_trn = mat.get("Y_trn").copy().astype(float)


X_tst = mat.get("X_tst").copy()
Y_tst = mat.get("Y_tst").copy()

#Constants
#Learnign Rate
LR = 0.000001
#Neurons in hidden level
#S_1 = 10
#Neurons in output level
S_2 = 3
#regularization constant
LAM = 0.01

def hadamard(X, Y):
    for i in range(X.shape[1]):
        X[:,i] = np.multiply(X[:,i], Y)
    return np.multiply(X, Y)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoidprime(Z):
    #dummy = sigmoid(Z).T * (1 - sigmoid(Z))
    #dummy = Z.T * (1 - Z)
    dummy = np.multiply(Z, (1 - Z))
    return dummy

def cost(A):
    argmax = np.max(A)
    if(argmax > 0.1):
        A[A < argmax] = 0
        A[A == argmax] = 1
    return A



del_W_1 = None
del_W_2 = None

for S_1 in [10, 20, 30, 50, 100]:

    # weight matrix
    # hidden layer
    W_1 = 0.5 * np.random.randn(S_1, X_trn.shape[1])
    # output layer
    W_2 = 0.5 * np.random.randn(S_2, S_1)

    # bias vectors
    # hidden layer
    B_1 = 0.01 * np.random.randn(S_1, 1)
    # output layer
    B_2 = 0.01 * np.random.randn(S_2, 1)



    while(True):
        for i in range(X_trn.shape[0]):
            Xi = np.mat(X_trn[i, :])

            #one hot encoding
            Yi = [i for i in range(S_2)]
            for j in range(S_2):
                if(j == Y_trn[i, :]):
                    Yi[j] = 1
                else:
                    Yi[j] = 0
            Yi = np.mat(Yi).T
            A_0 = Xi.T

            # foward propagation
            Z_1 = np.mat(np.dot(W_1, A_0)) + B_1
            A_1 = sigmoid(Z_1)
            Z_2 = np.mat(np.dot(W_2, A_1)) + B_2
            A_2 = sigmoid(Z_2)

            # backward propagation
            dummy = cost(A_2)
            #A_2 = dummy

            D_2 = np.multiply((cost(A_2) - Yi), sigmoidprime(Z_2))
            D_1 = np.multiply((np.mat(np.dot(W_2.T, D_2))), sigmoidprime(Z_1))

            del_W_1 = LR * np.mat(np.dot(D_1.T, Z_1)) + LAM * W_1
            del_W_2 = LR * np.mat(np.dot(D_2.T, Z_2)) + LAM * W_2

            #update weights and bias
            W_2 = W_2 - del_W_2
            B_2 = B_2 - LR * D_2
            W_1 = W_1 - del_W_1
            B_1 = B_1 - LR * D_1

        #check convergence by checking how small the delta is for each weight matrix
        if(np.all(del_W_1 < 0.000001) and np.all(del_W_2 < 0.000001)):
            break

    '''
    print("-----------------------------------training--------------------------------------------------------------------")
    for i in range(X_trn.shape[0]):
        Xi = np.mat(X_trn[i, :])
        Yi = [i for i in range(S_2)]
        for j in range(S_2):
            if(j == Y_trn[i, :]):
                Yi[j] = 1
            else:
                Yi[j] = 0
        Yi = np.mat(Yi).T

        A_0 = Xi.T
        # foward propagation
        Z_1 = np.mat(np.dot(W_1, A_0)) + B_1
        A_1 = sigmoid(Z_1)
        Z_2 = np.mat(np.dot(W_2, A_1)) + B_2
        A_2 = sigmoid(Z_2)

        print(Yi.T, " ", cost(A_2.T))
    '''
    print("#S_Hidden", S_1)
    print("---------------------------------------------------------------test---------------------------------------------")
    for i in range(X_tst.shape[0]):
        Xi = np.mat(X_tst[i, :])
        Yi = [i for i in range(S_2)]
        for j in range(S_2):
            if(j == Y_tst[i, :]):
                Yi[j] = 1
            else:
                Yi[j] = 0
        Yi = np.mat(Yi).T

        A_0 = Xi.T
        # foward propagation
        Z_1 = np.mat(np.dot(W_1, A_0)) + B_1
        A_1 = sigmoid(Z_1)
        Z_2 = np.mat(np.dot(W_2, A_1)) + B_2
        A_2 = sigmoid(Z_2)

        print(Yi.T, " ", cost(A_2.T))
