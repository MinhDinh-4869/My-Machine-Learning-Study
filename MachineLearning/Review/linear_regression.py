import random
import matplotlib.pyplot as plt
import numpy as np
#X = []
#y = []

#for i in range(20):
#    X.append([random.randint(0,50), 1])
#    y.append(X[len(X) -1][0]*3 + X[len(X) -1][1]*4 + random.random())

#X = np.array(X)
#y = np.array(y)

def load_data(file_name):
    file_handle = open(file_name, "r")
    lines = file_handle.readlines()
    X_train = []
    y_train = []
    for line in lines:
        temp = line.split(',')
        X_train.append([float(temp[i].strip()) for i in range(0, len(temp) - 1)] + [1])
        y_train.append(float(temp[len(temp) - 1].strip()))
    return np.array(X_train), np.array(y_train)

X_train, y_train = load_data("dat.txt")
y_train = y_train.reshape(len(y_train), 1)
X_size = X_train.shape

d = X_size[1]
N = X_size[0]

def LossFunct(ff, y_train):
    result= (ff - y_train).T.dot((ff - y_train)).flatten()
    return result[0]
def LinearReg(X_train, y_train, epochs, eta):
    w_init = np.random.random((1,d))
    #print(w_init)
    W = [w_init]
    loss = []
    for i in range(epochs):
        ff = X_train.dot(W[-1].T)
        loss.append(LossFunct(ff, y_train))

        dw = (ff - y_train).T.dot(X_train)  #e.T.dot(X)
        w_new = W[-1] - eta*dw
        if np.linalg.norm(w_new - W[-1]) < 1e-4:
            return W
        W.append(w_new)
    return W,loss

def LinearReg1(X_train, y_train):
    w = np.linalg.lstsq(X_train, y_train, rcond=None)
    return w

w = LinearReg1(X_train, y_train)
print(w[0])

W,loss = LinearReg(X_train, y_train, 500, 0.001)
thetas = W[-1]
print(thetas)
result = []
for i in range(20):
    result.append(thetas.dot(np.array([i, 1])))
plt.scatter(X_train[0:X_size[0], 0], y_train)
plt.plot(result, color = "r")
plt.show()
plt.plot(loss)
plt.show()
