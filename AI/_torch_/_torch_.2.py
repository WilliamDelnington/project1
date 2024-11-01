import numpy as np
import matplotlib.pyplot as plt

def f(x: np.array):
    return np.sin(x)

def df(x):
    return np.cos(x)

x = np.arange(-10, 10, 0.1)
y = f(x)

A = (3.14, f(3.14))

learning_rate = 0.01

for _ in range(500):
    x_new = A[0] - learning_rate * df(A[0])
    y_new = f(x_new)
    A = (x_new, y_new)

    plt.plot(x, y)
    plt.scatter(A[0], A[1], color="red")
    plt.pause(0.001)
    plt.clf()