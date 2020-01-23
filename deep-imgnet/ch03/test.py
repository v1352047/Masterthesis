import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return x[0]**2 + x[1]**2

x = np.array(np.arange(0.0, 20.0, 0.1), np.arange(0.0, 20.0, 0.1))

y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

