import numpy as np
import matplotlib.pyplot as plt
import math

#fx = x**2 + 10sinx



X = np.array([[0.50,0.7,1], [0.75,1.5,1], [1.00,1.25,1], [1.25,1.5,1], [1.50,2.0,1], [1.75,2.1,1], [1.75,0.2,1], [2.00,2.2,1], [2.25,1.0,1], [2.50,3.0,1], 
              [2.75,2,1], [3.00,1.5,1], [3.25,0.2,1], [3.50,3.7,1], [4.00,2.1,1], [4.25,3.0,1], [4.50,4.0,1], [4.75,3.0,1], [5.00,2.5,1], [5.50,5.0,1]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]).reshape(20,1)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def loss(w, x, y):
    zi = x.dot(w.T)
    return -1*(y[0][0]*math.log(zi[0][0]) + (1 - y[0][0])*math.log(1 - zi[0][0]))

def logistics_regression(X_train, y_train, epochs, eta):
    SIZE = X_train.shape
    w_init = np.random.random((1, SIZE[1]))
    W = [w_init]
    LOSS = []
    for i in range(epochs):
        x_id = np.random.permutation(SIZE[0])
        for j in x_id:
            xi = X_train[j, :].reshape(1, SIZE[1])
            yi = y_train[j, :].reshape(1,1)

            zi = xi.dot(W[-1].T)
            ai = sigmoid(zi[0]).reshape(1,1)

            #LOSS.append(loss(W[-1], xi, yi))

            dW = (ai - yi).T.dot(xi)
            w_new = W[-1] - eta*dW
            if (np.linalg.norm(w_new - W[-1])) < 1e-4:
                return W
            W.append(w_new)


def logistics_regression_copy(X_train, y_train, epochs, eta):
    SIZE = X_train.shape
    w_init = np.random.random((1, SIZE[1]))
    W = [w_init]
    for i in range(epochs):
        x_id = np.random.permutation(SIZE[0])
        for j in x_id:
            xi = X_train[j, :].reshape(1, SIZE[1])
            yi = y_train[j, :].reshape(1,1)

            z = xi.dot(W[-1].T)
            a = sigmoid(z[0]).reshape(1,1)

            dW = (a - yi).T.dot(xi)
            w_new = W[-1] - eta*dW

            if (np.linalg.norm(W[-1] - w_new)) < 1e-4:
                return W
            W.append(w_new)
    return W


W = logistics_regression(X, y, 500, 0.05)
print(W[-1])

thetas = W[-1][0]
x1 = np.linspace(0,10, 100)
x2= np.linspace(0,10, 100)


line = []
for i in x1:
  for j in x2:
    if sigmoid(thetas.dot(np.array([i, j, 1]))) > 0.8:
      line.append([i,j,1])

line = np.array(line)


plt.scatter(line[0:len(line), 0], line[0:len(line), 1],c = "y", marker =".")
plt.show()