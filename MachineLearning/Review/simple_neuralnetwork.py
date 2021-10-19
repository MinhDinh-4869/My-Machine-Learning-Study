import numpy as np
import matplotlib.pyplot as plt




#X = np.array([[1,1,1], [1,3,1], [3,1,1], [3,3,1]])
#y = np.array([1, 0, 0, 1]).reshape(4,1)


X = np.array([[0.50,0.7,1], [0.75,1.5,1], [1.00,1.25,1], [1.25,1.5,1], [1.50,2.0,1], [1.75,2.1,1], [1.75,0.2,1], [2.00,2.2,1], [2.25,1.0,1], [2.50,3.0,1], 
              [2.75,2,1], [3.00,1.5,1], [3.25,0.2,1], [3.50,3.7,1], [4.00,2.1,1], [4.25,3.0,1], [4.50,4.0,1], [4.75,3.0,1], [5.00,2.5,1], [5.50,5.0,1]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]).reshape(20,1)

C = 1
def sigmoid(z):
	return 1/(1 + np.exp(-z))

def logistics_regression1(X_train, y_train, epochs, eta):
	SIZE = X_train.shape
	w_init = np.random.random((C,SIZE[1] ))
	W = [w_init]
	for i in range(epochs):
		x_id = np.random.permutation(SIZE[0])
		for j in x_id:
			xi = X_train[j, :].reshape(1, SIZE[1])
			yi = y_train[j, :].reshape(1,1)

			zi = xi.dot(W[-1].T)
			ai = sigmoid(zi[0]).reshape(1,1)

			dW = (ai - yi).T.dot(xi)
			w_new = W[-1] - eta*dW
			if(np.linalg.norm(w_new - W[-1])) < 1e-4:
				return W
			W.append(w_new)
	return W

def logistics_regression2(X_train, y_train, epochs, eta):
	SIZE = X_train.shape
	w_init_1 = np.random.random((4,SIZE[1]))
	w_init_2 = np.random.random((4,4))
	w_init_3 = np.random.random((C, 4))
	W0 = [w_init_1]
	W1 = [w_init_2]
	W2 = [w_init_3]
	for i in range(epochs):
		x_id = np.random.permutation(SIZE[0])
		for j in x_id:
			xi = X_train[j, :].reshape(1, SIZE[1])
			yi = y_train[j, :].reshape(1,1)

			z1 = xi.dot(W0[-1].T)
			a1 = sigmoid(z1[0]).reshape(1,4)

			z2 = a1.dot(W1[-1].T)
			a2 = sigmoid(z2[0]).reshape(1,4)

			z3 = a2.dot(W2[-1].T)
			a3 = sigmoid(z3[0]).reshape(1,1)

			e2 = (a3 - yi)
			dW2 = e2.T.dot(a2)

			e1 = np.multiply((e2.dot(W2[-1])),np.multiply(a2,1 - a2))
			dW1 = e1.T.dot(a1)

			e0 = np.multiply((e1.dot(W1[-1])),np.multiply(a1,1 - a1))
			dW0 = e0.T.dot(xi)
			
			w_new_0 = W0[-1] - eta*dW0
			w_new_1 = W1[-1] - eta*dW1
			w_new_2 = W2[-1] - eta*dW2
			if(np.linalg.norm(w_new_1 - W1[-1])) < 1e-4 and (np.linalg.norm(w_new_0 - W0[-1])) < 1e-4 and (np.linalg.norm(w_new_2 - W2[-1])) < 1e-4:
				return W0, W1, W2
			W0.append(w_new_0)
			W1.append(w_new_1)
			W2.append(w_new_2)
	return W0, W1, W2


#Stochaic Gradient Descent
def logistics_regression(X_train, y_train, epochs, eta):
	SIZE = X_train.shape
	w_init_1 = np.random.random((4,SIZE[1]))
	w_init_2 = np.random.random((4,4))
	w_init_3 = np.random.random((C, 4))
	W0 = [w_init_1]
	W1 = [w_init_2]
	W2 = [w_init_3]
	for i in range(epochs):
		x_id = np.random.permutation(SIZE[0])
		for j in x_id:
			xi = X_train[j, :].reshape(1, SIZE[1])
			yi = y_train[j, :].reshape(1,1)


			z1 = xi.dot(W0[-1].T)
			a1 = sigmoid(z1[0]).reshape(1,4)


			z2 = a1.dot(W1[-1].T) + 1
			a2 = sigmoid(z2[0]).reshape(1,4)


			z3 = a2.dot(W2[-1].T) + 1
			a3 = sigmoid(z3[0]).reshape(1,1)


			e2 = (a3 - yi)
			dW2 = e2.T.dot(a2)

			e1 = np.multiply((e2.dot(W2[-1])),np.multiply(a2,1 - a2))
			dW1 = e1.T.dot(a1)

			e0 = np.multiply((e1.dot(W1[-1])),np.multiply(a1,1 - a1))
			dW0 = e0.T.dot(xi)
			
			w_new_0 = W0[-1] - eta*dW0
			w_new_1 = W1[-1] - eta*dW1
			w_new_2 = W2[-1] - eta*dW2
			if(np.linalg.norm(w_new_1 - W1[-1])) < 1e-4 and (np.linalg.norm(w_new_0 - W0[-1])) < 1e-4 and (np.linalg.norm(w_new_2 - W2[-1])) < 1e-4:
				return W0, W1, W2
			W0.append(w_new_0)
			W1.append(w_new_1)
			W2.append(w_new_2)
	return W0, W1, W2



W1, W2, W3 = logistics_regression(X, y, 500, 0.05)
print(W1[-1], W2[-1], W3[-1])


x1 = np.linspace(0,10, 50)
x2= np.linspace(0,10, 50)


line = []
for i in x1:
	for j in x2:
		xi = np.array([i,j, 1]).reshape(1, 3)

		z1 = xi.dot(W1[-1].T)
		a1 = sigmoid(z1[0]).reshape(1,4)

		z2 = a1.dot(W2[-1].T) + 1
		a2 = sigmoid(z2[0]).reshape(1,4)

		z3 = a2.dot(W3[-1].T) + 1
		a3 = sigmoid(z3[0]).reshape(1,1)
		print(a3)
		if a3[0][0] > 0.8:
			line.append([i,j,1])

line = np.array(line)


plt.scatter(line[0:len(line), 0], line[0:len(line), 1],c = "y", marker =".")
for i in range(len(X)):
	if(y[i][0] == 1):
		plt.scatter(X[i, 0], X[i, 1], c="r", marker="o")
	else:
		plt.scatter(X[i, 0], X[i, 1], c="b", marker="o")
plt.show()