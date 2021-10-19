import numpy as np
import matplotlib.pyplot as plt
import math

#fx = x**2 + 10sinx

def grad(x):
    return 2*x + 10*math.cos(x)
def F(x):
    return x**2 + 10*math.sin(x)

def gradient_descent(x_init, eta):
    X = [x_init]
    while True:
        x_new = X[-1] - eta*grad(X[-1])
        if (np.linalg.norm(x_new - X[-1])) < 1e-4:
            return X
        X.append(x_new)


def momentum_gradient_descent(x_init, eta, gamma):
    X = [x_init]
    v_old = 0
    while True:
        v_new = gamma*v_old + eta*grad(X[-1])
        x_new = X[-1] - v_new
        if (np.linalg.norm(x_new - X[-1])) < 1e-4:
            return X
        X.append(x_new)
        v_old = v_new

def nesterov_gradient_descent(x_init, eta, gamma):
    X = [x_init]
    v_old = 0
    while True:
        v_new = gamma*v_old + eta*grad(X[-1] - gamma*v_old)
        x_new = X[-1] - v_new
        if (np.linalg.norm(x_new - X[-1])) < 1e-4:
            return X
        X.append(x_new)
        v_old = v_new
x_init = np.random.randint(-5, 10)
X = nesterov_gradient_descent(x_init, 0.05, 0.8)


plt.axhline(y=0.0, color='r', linestyle='-')
plt.axvline(x=0.0, color='r', linestyle='-')

horz = np.linspace(-5, 10, 200)
line = []
for i in horz:
    line.append(F(i))

dot = []
for i in X:
    dot.append(F(i))

plt.plot(horz, line, 'k.')
plt.plot(X, dot, 'ro')
plt.plot(X[-1], dot[-1], 'bo')
plt.show()
