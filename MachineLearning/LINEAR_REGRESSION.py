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
n = len(X_train[0])
m = len(X_train)
def LinearRegression(Xtrain, yTrain, eta, epochs):
    thetas = np.random.random(n)
    loss = []
    for i in range(epochs):
        feed_forward = thetas.dot(Xtrain.T)
        loss_val = 0.5*(1/m)*((yTrain - feed_forward).dot(yTrain - feed_forward))
        loss.append(loss_val)
        thetas = thetas - eta*((feed_forward - yTrain).dot(Xtrain))
    return thetas, loss

thetas, loss = LinearRegression(X_train, y_train, 0.0001, 500)
print(thetas)
result = []
for i in range(20):
    result.append(thetas.dot(np.array([i, 1])))
plt.scatter(X_train[0:m, 0], y_train)
plt.plot(result, color = "r")
plt.show()
plt.plot(loss)
plt.show()
