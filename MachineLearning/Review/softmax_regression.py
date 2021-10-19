import numpy as np
import matplotlib.pyplot as plt


means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
d = 2#dimension of data
C = 3#number of classes
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

# each column is a datapoint
X = np.concatenate((X0, X1, X2), axis = 0)
# extended data
y = np.full((1500, 3), 0)
y[0:500, 0] = 1
y[500:1000, 1] = 1
y[1000:1500, 2] = 1
print(y)
C = 3
X = X.T
X = np.concatenate( (X, np.ones((1, 3*N))) , axis = 0)
X = X.T



def stable_softmax(z):
    c = max(z)
    a = np.exp(z - c)
    return a/(np.sum(a))
#suppose we have Xtrain, ytrain.
C = 3

def softmax_reg(X_train, y_train, epochs, eta):
    SIZE = X_train.shape
    print(SIZE)
    w_init = np.random.random((C, SIZE[1]))
    W = [w_init]
    check_after = 20
    count = 0
    for i in range(epochs):
        x_id = np.random.permutation(SIZE[0])
        for j in x_id:
            xi = X_train[j ,:].reshape(1, SIZE[1])
            yi = y[j, :].reshape(1, C)
            zi = xi.dot(W[-1].T)
            ai = stable_softmax(zi[0]).reshape(1,C)
            #print(ai)

            dW = (ai - yi).T.dot(xi)
            w_new = W[-1] - eta*dW
            count+=1
            if count % check_after == 0:
                if np.linalg.norm(w_new - W[-check_after]) < 1e-4:
                    return W
            W.append(w_new)
    return W


def softmax_regression(X_train, Y_train, epochs, eta):
    w_init = np.random.random((C, X_train.shape[1]))
    W = [w_init]
    check_after = 20
    count = 0
    for i in range(epochs):
        x_index = np.random.permutation(X_train.shape[0])
        for j in x_index:
            xi = X_train[j, :].reshape(1, X_train.shape[1])
            yi = Y_train[j, :].reshape(1, C)

            zi = xi.dot(W[-1].T)
            ai = stable_softmax(zi[0]).reshape(1, C)

            dW = (ai - yi).T.dot(xi)
            w_new = W[-1] - eta*dW
            count+=1
            if count % check_after == 0:
                if np.linalg.norm(w_new - W[-check_after]) < 1e-4:
                    return W
            W.append(w_new)
    return W

W = softmax_regression(X, y, 500, 0.05)
print(W[-1])


def pred(w,x):
    """
    predict output of each columns of X
    Class of each x_i is determined by location of max probability
    Note that class are indexed by [0, 1, 2, ...., C-1]
    """
    #print(x)
    z = x.dot(w.T) #+ 1
    a = stable_softmax(z)
    #print(a)
    return np.argmax(a, axis = 0)




x1 = np.linspace(-1,10, 50)
x2= np.linspace(-1,10,50)

for i in x1:
    for j in x2:
        point = np.array([i,j, 1])
        label = pred(W[-1], np.array(point))
        #print(label,"\t", [i,j, 1])
        if label ==0:
            plt.scatter(point[0], point[1],c = "b", marker =".")
        elif label == 1:
            plt.scatter(point[0], point[1],c = "y", marker =".")
        else:
            plt.scatter(point[0], point[1],c = "r", marker =".")

plt.scatter(X[0:N, 0], X[0:N, 1],c = "r", marker ="o")
plt.scatter(X[N:2*N, 0], X[N:2*N, 1],c = "b", marker ="s")
plt.scatter(X[2*N:3*N, 0], X[2*N:3*N, 1],c = "y", marker ="^")
plt.show()