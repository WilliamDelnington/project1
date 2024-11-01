import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return np.sin(5 * x) * np.cos(5 * y) / 5

def df(x, y):
    return np.cos(5 * x) * np.cos(5 * y), -np.sin(5 * x) * np.sin(5 * y)

x = np.arange(-1, 1, 0.05)
y = np.arange(-1, 1, 0.05)

X, Y = np.meshgrid(x, y)

Z = f(X, Y)

A = (0.7, 0.4, f(0.7, 0.4))
B = (0.3, 0.8, f(0.3, 0.8))
C = (-0.5, 0.5, f(-0.5, 0.5))

learning_rate = 0.01

ax = plt.subplot(projection="3d", computed_zorder=False)

for _ in range(900):
    dfx, dfy = df(A[0], A[1])
    X_new, Y_new = A[0] - learning_rate * dfx, A[1] - learning_rate * dfy
    A = (X_new, Y_new, f(X_new, Y_new))

    dfx, dfy = df(B[0], B[1])
    X_new, Y_new = B[0] - learning_rate * dfx, B[1] - learning_rate * dfy
    B = (X_new, Y_new, f(X_new, Y_new))

    dfx, dfy = df(C[0], C[1])
    X_new, Y_new = C[0] - learning_rate * dfx, C[1] - learning_rate * dfy
    C = (X_new, Y_new, f(X_new, Y_new))

    ax.plot_surface(X, Y, Z, cmap="viridis", zorder=0)
    ax.scatter(A[0], A[1], A[2], color="red", zorder=1)
    ax.scatter(B[0], B[1], B[2], color="green", zorder=1)
    ax.scatter(C[0], C[1], C[2], color="cyan", zorder=1)
    plt.pause(0.01)
    ax.clear()

plt.show()