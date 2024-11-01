import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("E:/Works/AI/_torch_/df_train.csv")

data_filter = data[["price", "living_in_m2"]]

def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].price
        y = points.iloc[i].living_in_m2
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points)) 

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].price / 20000
        y = points.iloc[i].living_in_m2 / 10

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

m = 0
b = 0
# Learning rate
L = 0.001

epochs = 300

data_filter = data_filter.head(40)

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data_filter, L)

print(m, b)

plt.scatter(data_filter.price, data_filter.living_in_m2, color='black')

plt.plot(list(range(100, 700)), [m * x + b for x in range(100, 700)], color='red')
plt.show()