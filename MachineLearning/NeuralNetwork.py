import math
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0.50,0.7], [0.75,1.5], [1.00,1.25], [1.25,1.5], [1.50,2.0], [1.75,2.1], [1.75,0.2], [2.00,2.2], [2.25,1.0], [2.50,3.0], 
              [2.75,2], [3.00,1.5], [3.25,0.2], [3.50,3.7], [4.00,2.1], [4.25,3.0], [4.50,4.0], [4.75,3.0], [5.00,2.5], [5.50,5.0]])
y = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]])
bias = np.full((len(X)), 1)
def Sigmoid(x):
    return 1/(1 + np.exp(-x))

def has_converged(loss):
    return loss < 1
def loss_funct(y, feedforward):
    result = np.dot(y.flatten(), np.log(feedforward.flatten())) + np.dot((1 - y.flatten()),np.log(1 - feedforward.flatten()))
    return result*-1
def FF(X_train, w0, w1, w2):
    b0 = np.full((len(X_train),4), 1)
    b1 = np.full((len(X_train),4), 1)
    b2 = np.full((len(X_train),1), 1)
    
    z1 = X_train.dot(w0.T) + b0
    a1 = Sigmoid(z1)
    z2 = a1.dot(w1.T) + b1
    a2 = Sigmoid(z2)
    z3 = a2.dot(w2.T) + b2
    a3 = Sigmoid(z3)    
    return a3.flatten()    

def NeuralNet(X_train ,Y_train,epochs, eta):
    w0 = np.random.random((4,2))
    b0 = np.full((20,4), 1)
    w1 = np.random.random((4,4))
    b1 = np.full((20,4), 1)
    w2 = np.random.random((1,4))
    b2 = np.full((20,1), 1)

    #The size of the weights matrix is m x n, where n is the dimension of the input layer and m is the dimention of the next layer
    #The bias matrix is of size N x m, where N is the number of input points

    loss = []
    #while True:
    for i in range(epochs):
        z1 = X_train.dot(w0.T) + b0
        a1 = Sigmoid(z1)

        z2 = a1.dot(w1.T) + b1
        a2 = Sigmoid(z2)

        z3 = a2.dot(w2.T) + b2
        a3 = Sigmoid(z3)

        loss.append(loss_funct(Y_train, a3))

        e2 = (a3 - Y_train)
        dw2 = e2.T.dot(a2) #dW = e.T.dot(a)

        e1 = np.multiply((e2.dot(w2)),np.multiply(a2,1 - a2)) #(a2 - Y_train)
        dw1 = e1.T.dot(a1) #dW = e.T.dot(a)

        e0 = np.multiply((e1.dot(w1)),np.multiply(a1,1 - a1))  #er = er+1.dot(wr+1).multiply(grad(zr+1) == ar+1(1 - ar+1)) 
        dw0 = e0.T.dot(X_train)

        w2 = w2 - eta*dw2
        w1 = w1 - eta*dw1
        w0 = w0 - eta*dw0
        
    return w0, w1, w2, loss

w0, w1,w2, loss= NeuralNet(X, y.T,1500, 0.05)
print(w0)
print(w1)
print(w2)
plt.plot(loss)
plt.show()
x1 = np.linspace(0,10, 100)
x2= np.linspace(0,10, 100)

#print(x)
mask = y.flatten() > 0
mask_1 = (y.flatten() == 0)
yes = X[mask]
no = X[mask_1]
line = []
for i in x1:
  for j in x2:
    if FF(np.array([[i,j]]), w0, w1, w2) > 0.8:
      line.append([i,j])

line = np.array(line)

test = np.array([[4.15,3.2], [4.20,4.1], [4.75,3.15], [5.10,2.2], [4.10,5.2], [8.10,3.2], [10.10,6.2]])
out_put = FF(test, w0, w1, w2)
print(out_put)
mask_2= out_put > 0.8
mask_3 = out_put < 0.8
yes_test = test[mask_2]
no_test = test[mask_3]
#print(yes_test)
#print(no_test)
#print(line)
plt.scatter(line[0:len(line), 0], line[0:len(line), 1],c = "y", marker =".")
plt.scatter(yes[0:len(yes), 0], yes[0:len(yes), 1],c = "r", marker ="o")
plt.scatter(yes_test[0:len(yes_test), 0], yes_test[0:len(yes_test), 1],c = "r", marker ="s")
plt.scatter(no[0:len(no), 0], no[0:len(no), 1],c = "b", marker ="o")
plt.scatter(no_test[0:len(no_test), 0], no_test[0:len(no_test), 1],c = "b", marker ="s")

plt.show()