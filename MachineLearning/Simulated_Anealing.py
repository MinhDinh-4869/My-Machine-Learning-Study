import numpy as np
import matplotlib.pyplot as plt
import random
import math

def H(position):
    return position**2 + 10*math.sin(position)

def Simulated_Anealing(steps):
    init_pos =10# random.randint(-5, 10)
    track = [init_pos]
    pos = init_pos
    for step in range(steps):
        T = 2*max(0, ((steps-step*1.2)/steps))**3
        new_pos = pos + random.randint(-10,10) + random.random()
        S_old = H(pos)
        S_new = H(new_pos)
        if S_new < S_old:
            pos = new_pos
            track.append(pos)
        else:
            if T == 0:
                accepted_prob = 0
            else:
                accepted_prob = math.exp((S_old - S_new)/T)
                if  random.random() < accepted_prob:
                    pos = new_pos
                    track.append(pos)
    return track

theta = Simulated_Anealing(50)
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
plt.show()