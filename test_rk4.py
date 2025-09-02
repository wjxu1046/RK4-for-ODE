import torch
import numpy as np


def u_exact(t):
    return torch.exp(-0.5 * t ** 2)

def f(t, y):
    return -t * y

def RK4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6.0 * (k1 + 2 * k2 + 2* k3 + k4)

def RK4_solve(f, t0, t1, y0, N, a, b):
    h = (b -a) / N
    t = t0
    ys = []
    
    while t < t1:
        h1 = min(t1 - t, h)
        y1 = RK4_step(f, t, y0, h1)
        y0 = y1
        t = t + h1
        ys.append(y1)
        
    
    return ys


a, b = 0, 1
t0, t1 = 0 , 1
y0 = 1
N = 320

t = torch.linspace(t0, t1, N)
u_pre = RK4_solve(f, t0, t1, y0, N, a, b)
u_true = u_exact(t)


import matplotlib.pyplot as plt

u_true = u_true.numpy()
error = np.max(np.abs(u_pre - u_true))
print(error)

plt.figure()
plt.plot(t,u_pre,'r',label = 'pre solution')
plt.plot(t,u_true,'b',label = 'true solution')
plt.title('RK4 for solution')
plt.legend()
plt.show()
