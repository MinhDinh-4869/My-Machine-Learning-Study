import numpy as np 
import matplotlib.pyplot as plt


def softmax(Z):
    """
    Compute softmax values for each sets of scores in V.
    each column of V is a set of score.    
    """
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis = 0)
    return A

# randomly generate data 
N = 2 # number of training sample 
d = 2 # data dimension 
C = 3 # number of classes 

X = np.random.randn(d, N)
y = np.random.randint(0, 3, (N,))

#---------------------------------------------------------------------- 

## One-hot coding
from scipy import sparse 
def convert_labels(y, C = C):
    """
    convert 1d label to a matrix label: each column of this 
    matrix coresponding to 1 element in y. In i-th column of Y, 
    only one non-zeros element located in the y[i]-th position, 
    and = 1 ex: y = [0, 2, 1, 0], and 3 classes then return

            [[1, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 1, 0, 0]]
    """
    Y = sparse.coo_matrix((np.ones_like(y), 
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y 
#-----------------------------------------------------------------------


Y = convert_labels(y, C)

# cost or loss function  
def cost(X, Y, W):
    A = softmax(W.T.dot(X))
    return -np.sum(Y*np.log(A))

W_init = np.random.randn(d, C)

def grad(X, Y, W):
    A = softmax((W.T.dot(X)))
    E = A - Y
    return X.dot(E.T) # grad = X.E = X.(y^ - y)
    
def numerical_grad(X, Y, W, cost):
    eps = 1e-6
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_p = W.copy()
            W_n = W.copy()
            W_p[i, j] += eps 
            W_n[i, j] -= eps
            g[i,j] = (cost(X, Y, W_p) - cost(X, Y, W_n))/(2*eps)
    return g 


def softmax_regression(X, y, W_init, eta, tol = 1e-4, max_count = 10000):
    W = [W_init]    
    C = W_init.shape[1]
    Y = convert_labels(y, C)
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    
    count = 0
    check_w_after = 20
    while count < max_count:
        # mix data 
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = Y[:, i].reshape(C, 1)
            ai = softmax(np.dot(W[-1].T, xi)) #W[-1].T, xi = z1, softmax(z1) = ai1
            W_new = W[-1] + eta*xi.dot((yi - ai).T)
            count += 1
            # stopping criteria
            if count%check_w_after == 0:                
                if np.linalg.norm(W_new - W[-check_w_after]) < tol:
                    print(count)
                    return W
            W.append(W_new)
    return W


def pred(W, X):
    """
    predict output of each columns of X
    Class of each x_i is determined by location of max probability
    Note that class are indexed by [0, 1, 2, ...., C-1]
    """
    A = softmax(W.T.dot(X))
    return np.argmax(A, axis = 0)

#--------------------------------------------------create data
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

# each column is a datapoint
X = np.concatenate((X0, X1, X2), axis = 0).T 
# extended data
X = np.concatenate((np.ones((1, 3*N)), X), axis = 0)
C = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T

#------------------------------------------------------end

W_init = np.random.randn(X.shape[0], C)
W = softmax_regression(X, original_label, W_init, .05)
print(W[-1])
print(X)


x1 = np.linspace(0,10, 50)
x2= np.linspace(0,10, 50)

for i in x1:
    for j in x2:
        point = np.array([1,i,j])
        label = pred(W[-1], np.array(point).T)
        #print(label,"\t", [i,j])
        if label ==0:
            plt.scatter(point[1], point[2],c = "r", marker =".")
        elif label == 1:
            plt.scatter(point[1], point[2],c = "b", marker =".")
        else:
            plt.scatter(point[1], point[2],c = "y", marker =".")

plt.scatter(X[1, 0:N], X[2, 0:N],c = "r", marker ="o")
plt.scatter(X[1, N:2*N], X[2, N:2*N],c = "b", marker ="s")
plt.scatter(X[1, 2*N:3*N], X[2, 2*N:3*N],c = "y", marker =".")
plt.show()