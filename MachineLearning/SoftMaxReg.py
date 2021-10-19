import numpy as np
import math
import matplotlib.pyplot as plt


def StableSoftMax(array):
	c = max(array)
	s = np.sum(np.exp(array - c))
	return np.exp(array - c)/s

def softmax(Z):
	"""
	Compute softmax values for each sets of scores in V.
	each column of V is a set of score.    
	"""
	e_Z = np.exp(Z)
	A = e_Z / e_Z.sum(axis = 0)
	return A

def SingleSoftMax(z):
	zout = np.copy(z)
	for i in range(len(z)):
		zout[i] = StableSoftMax(z[i])
	return zout

def grad(x, y, a):
	return (a - y).T.dot(x)

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


def softmaxregression(X, y, eta, epochs):
	w = np.random.random((C, d + 1))
	n = X.shape
	count = 0
	check_after = 5
	w_copare = w
	for i in range(epochs):
		x_id = np.random.permutation(n[0])
		#Using stochaic gradient descent
		for j in x_id:
			xi = X[j].reshape(1,d+1)
			yi = y[j].reshape(1, C)
			a = np.array([StableSoftMax(xi[0].dot(w.T))])
			w = w - eta*((a - yi).T.dot(xi))
			count+=1
			if count % check_after == 0:
				if np.linalg.norm(w - w_copare) < 1e-4:
					print("loop no.", i)
					return w
				w_copare = w

	return w


def softmaxregression_1(X, y, eta, epochs):
	w_init = np.random.random((C, d + 1))
	W = [w_init]
	#b = np.full((1500, C), 1)
	n = X.shape
	count = 0
	check_after = 1
	for i in range(epochs):
			#Using normal gradient descent
			z = X.dot(W[-1].T)
			a = SingleSoftMax(z)
			dW = (a - y).T.dot(X)   #dw = E.T.dot(A)
			w_new = W[-1] - eta*dW
			#dw = (a1 - y[j]).reshape(1,C).T.dot(X[j].reshape(1,d+1))			
			count+=1
			if count % check_after == 0:
				print(count)
				if np.linalg.norm(w_new - W[-check_after]) < 1e-4:
					print("loop no.", count)
					return W
			W.append(w_new)
	return W

W_ = softmaxregression_1(X, y, 0.05, 500)
W = W_[-1]
#W = np.array([[-0.56161943 , 0.83297992],
# [ 1.66292153 ,-0.74470617],
# [ 0.33316646 , 1.8066097 ]])
print(W)
#w = np.random.random((C, 2))
b = np.full((1500, C), 1)



def pred(w,x):
	"""
	predict output of each columns of X
	Class of each x_i is determined by location of max probability
	Note that class are indexed by [0, 1, 2, ...., C-1]
	"""
	#print(x)
	z = x.dot(w.T) #+ 1
	a = StableSoftMax(z)
	#print(a)
	return np.argmax(a, axis = 0)




x1 = np.linspace(-1,10, 50)
x2= np.linspace(-1,10, 50)

for i in x1:
	for j in x2:
		point = np.array([i,j, 1])
		label = pred(W, np.array(point))
		#print(label,"\t", [i,j, 1])
		if label ==0:
			plt.scatter(point[0], point[1],c = "k", marker =".")
		elif label == 1:
			plt.scatter(point[0], point[1],c = "b", marker =".")
		else:
			plt.scatter(point[0], point[1],c = "y", marker =".")

plt.scatter(X[0:N, 0], X[0:N, 1],c = "r", marker ="o")
plt.scatter(X[N:2*N, 0], X[N:2*N, 1],c = "b", marker ="s")
plt.scatter(X[2*N:3*N, 0], X[2*N:3*N, 1],c = "y", marker ="^")
plt.show()






			#z1 = X[j].dot(w.T) #+ 1# + b
			#a1 = StableSoftMax(z1)
			#a1 = softmax(z1)