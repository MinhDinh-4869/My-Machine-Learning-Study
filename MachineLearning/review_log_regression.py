import math
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0.50,0.7], [0.75,1.5], [1.00,1.25], [1.25,1.5], [1.50,2.0], [1.75,2.1], [1.75,0.2], [2.00,2.2], [2.25,1.0], [2.50,3.0], 
              [2.75,2], [3.00,1.5], [3.25,0.2], [3.50,3.7], [4.00,2.1], [4.25,3.0], [4.50,4.0], [4.75,3.0], [5.00,2.5], [5.50,5.0]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
bias = np.full((len(X)), 1)
def Sigmoid(x):
    return 1/(1 + np.exp(-x))

def has_converged(loss):
    return loss < 1e+3
def loss_funct(y, feedforward):
    result = np.dot(y, np.log(feedforward)) + np.dot((1 - y),np.log(1 - feedforward))
    return result*-1
def logistics_reg(X_train ,Y_train,epochs, eta):
    w = np.random.random((len(X_train[0])))
    loss = []
    #while True:
    for i in range(epochs):
        feedforward = Sigmoid(w.dot(X_train.T) + bias)
        loss.append(loss_funct(Y_train, feedforward))
        ID = np.random.permutation(len(X_train))
        w_old = np.copy(w)
        for i in ID:
            w = w  + eta*(Y_train[i] - feedforward[i])*X_train[i]
    return w, loss

thetas, loss= logistics_reg(X, y,100, 0.05)
print(thetas)
plt.plot(loss)
plt.show()
x1 = np.linspace(0,10, 100)
x2= np.linspace(0,10, 100)

#print(x)
mask = y > 0
mask_1 = (y == 0)
yes = X[mask]
no = X[mask_1]
line = []
for i in x1:
  for j in x2:
    if Sigmoid(thetas.dot(np.array([i, j])) + 1) > 0.8:
      line.append([i,j])

line = np.array(line)

test = np.array([[4.15,3.2], [4.20,4.1], [4.75,3.15], [5.10,2.2], [5.10,5.2]])
out_put = Sigmoid(thetas.dot(test.T) + np.full((5), 1))  
print(out_put)
mask_2= out_put > 0.8
mask_3 = out_put < 0.8
yes_test = test[mask_2]
no_test = test[mask_3]
print(yes_test)
print(no_test)
print(line)
plt.scatter(line[0:len(line), 0], line[0:len(line), 1],c = "y", marker =".")
plt.scatter(yes[0:len(yes), 0], yes[0:len(yes), 1],c = "r", marker ="o")
plt.scatter(no[0:len(no), 0], no[0:len(no), 1],c = "b", marker ="o")

plt.show()