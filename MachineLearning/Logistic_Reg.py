import numpy as np
import matplotlib.pyplot as plt
def Sigmoid(x):
    return 1/(1 + np.exp(-x))
def LossFunction(feed_forward,y):
  result = (y.dot(np.log(feed_forward))) + ((1 - y).dot(np.log(1 - feed_forward)))
  return result*-1
#def Has_converged()


X = np.array([[0.50,0.7,1], [0.75,1.5,1], [1.00,1.25,1], [1.25,1.5,1], [1.50,2.0,1], [1.75,2.1,1], [1.75,0.2,1], [2.00,2.2,1], [2.25,1.0,1], [2.50,3.0,1], 
              [2.75,2,1], [3.00,1.5,1], [3.25,0.2,1], [3.50,3.7,1], [4.00,2.1,1], [4.25,3.0,1], [4.50,4.0,1], [4.75,3.0,1], [5.00,2.5,1], [5.50,5.0,1]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

def Logistic_Reg(X_train, Y_train, epochs, eta):
    theta = np.random.random(len(X_train[0]))
    loss = []
    for i in range(epochs):
        feed_forward = Sigmoid(theta.dot(X_train.T))
        loss.append(LossFunction(feed_forward, Y_train))
        ID = np.random.permutation(len(X_train))
        for i in ID:
            theta = theta - eta*(feed_forward[i] - Y_train[i])*X_train[i]
    return theta

thetas = (Logistic_Reg(X, y, 100, 0.05))
print(thetas)
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
    if Sigmoid(thetas.dot(np.array([i, j, 1]))) > 0.8:
      line.append([i,j,1])

line = np.array(line)
test = np.array([[4.15,3.2,1], [4.20,4.1,1], [4.75,3.15,1], [5.10,2.2,1], [5.10,5.2,1]])
out_put = Sigmoid(thetas.dot(test.T))
print(out_put)
mask_2= out_put > 0.8
mask_3 = out_put < 0.8
yes_test = test[mask_2]
no_test = test[mask_3]
print(yes_test)
print(no_test)

plt.scatter(line[0:len(line), 0], line[0:len(line), 1],c = "y", marker =".")
plt.scatter(yes[0:len(yes), 0], yes[0:len(yes), 1],c = "r", marker ="o")
plt.scatter(no[0:len(no), 0], no[0:len(no), 1],c = "b", marker ="o")

plt.show()