import random
import numpy as np
import matplotlib.pyplot as plt
import math

#function f(x) = x2 + 10sinx


def grad(x):
    return 2*x + 10*math.cos(x)

def has_converged(x_new):
    return np.linalg.norm(grad(x_new))/(1) < 1e-3

def grad_descent(x_init, eta):
    X = [x_init]
    while True:
        x_new = X[-1] - eta*grad(X[-1])
        if has_converged(x_new):
            break
        X.append(x_new)
    return X
def momentum_grad_descent(x_init, eta, gamma):
    X= [x_init]
    v_old = 0
    while True:
        v_new = gamma*v_old + eta*grad(X[-1])
        x_new = X[-1] - v_new
        if(has_converged(x_new)):
            break
        X.append(x_new)
        v_old = v_new
    return X
def nesterov_grad_descent(x_init, eta, gamma):
    X =[x_init]
    v_old = 0
    while True:
        v_new = gamma*v_old + eta*grad(X[-1] - gamma*v_old)
        x_new = X[-1] - v_new
        if has_converged(x_new):
            break
        X.append(x_new)
        v_old = v_new
    return X
theta = nesterov_grad_descent(10, 0.05, 0.8)
#graph
F = []
x = -5
X_ = []
while x < 10:
    F.append(x**2 + 10*math.sin(x))
    X_.append(x)
    x = x + 0.5


PATH = []
for i in theta:
    PATH.append(i**2 + 10*math.sin(i))
plt.plot(X_, F)
plt.plot(theta, PATH, 'ko')
plt.plot(theta[-1], PATH[-1], 'ro')
plt.show()