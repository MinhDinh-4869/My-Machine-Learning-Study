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
def FF(X_train, W):
	b0 = np.full((len(X_train),4), 1)
	b1 = np.full((len(X_train),4), 1)
	b2 = np.full((len(X_train),1), 1)
	b = [b0, b1, b2]
	a = [X_train, 0, 0, 0] #size W + 1
	z = [0, 0, 0, 0] #size = size of W
	for i in range(1,len(a)):
		z[i] = a[i-1].dot(W[i-1].T) + b[i - 1] #z1
		a[i] = Sigmoid(z[i])
	return a[-1].flatten()  

def NeuralNet(X_train ,Y_train,epochs, eta):
	#w size = (next layer size, current layer size)
	#b size = (X train size, w size[0])
	w0 = np.random.random((4,2))
	b0 = np.full((len(X_train),4), 1)
	w1 = np.random.random((4,4))
	b1 = np.full((len(X_train),4), 1) #size b = size a x size w_next.T
	w2 = np.random.random((1,4))
	b2 = np.full((len(X_train),1), 1)


	W = [w0, w1, w2]
	b = [b0, b1, b2]
	a = [X_train, 0, 0, 0] #size W + 1
	z = [0, 0, 0, 0] #size = size of W
	e = [0, 0, 0]
	dW = [0, 0, 0]
	loss = []
	#while True:
	for loop in range(epochs):
		for i in range(1,len(a)):
			z[i] = a[i-1].dot(W[i-1].T) + b[i - 1] #z1
			a[i] = Sigmoid(z[i])

		loss.append(loss_funct(Y_train, a[-1]))


		e[-1] = (a[-1] - Y_train)
		dW[-1] = e[-1].T.dot(a[-2]) #dW = e.T.dot(a)

		count  = len(a) - 3
		while count >= 0:
			e[count] = np.multiply((e[count + 1].dot(W[count + 1])),np.multiply(a[count + 1],1 - a[count + 1]))  #er = er+1.dot(wr+1).multiply(grad(zr+1) == ar+1(1 - ar+1)) 
			dW[count] = e[count].T.dot(a[count])
			count = count - 1

		for i in range(len(W)):
			W[i] = W[i] - eta*dW[i]
		
	return W, loss

W, loss= NeuralNet(X, y.T,1500, 0.05)
print(W)


plt.plot(loss)
plt.show()
x1 = np.linspace(0,10, 300)
x2= np.linspace(0,10, 300)

#print(x)
mask = y.flatten() > 0
mask_1 = (y.flatten() == 0)
yes = X[mask]
no = X[mask_1]
line = []
for i in x1:
	for j in x2:
			if FF(np.array([[i,j]]), W) > 0.8:
				line.append([i,j])

line = np.array(line)

test = np.array([[4.15,3.2], [4.20,4.1], [4.75,3.15], [5.10,2.2], [4.10,5.2], [8.10,3.2], [10.10,6.2]])
out_put = FF(test, W)
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