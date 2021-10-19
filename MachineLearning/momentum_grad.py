import numpy as np
import matplotlib.pyplot as plt
import random
import math

def grad(theta):
    return theta*2 + 10*math.cos(theta)
def has_converged(theta_new):
    return np.linalg.norm(grad(theta_new))/(1) < 1e-3
def gradient_des_momentum(theta_init, gamma, eta):
    theta = [theta_init] #keep track on the position, theta init = 0, f(theta)
    v_old = np.zeros_like(theta_init)
    #while True:
    for i in range(100):
        #v_new based on the current v (v_old) and the "v riêng" (F'(theta))
        v_new = gamma*v_old + eta*grad(theta[-1])# - gamma*v_old)#(2*theta[-1] + cos(theta[-1]))
        theta_new = theta[-1] - v_new
        if has_converged(theta_new):
            break
        theta.append(theta_new)
        v_old = v_new
    return theta

theta = (gradient_des_momentum(10, 0.9, 0.1))

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