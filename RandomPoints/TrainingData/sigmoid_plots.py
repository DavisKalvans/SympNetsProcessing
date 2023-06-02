import numpy as np
import matplotlib.pyplot as plt

# Plotting options
font = {'family' : 'serif',
        'size'   : 16}
plt.rc('font', **font)

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def sigmoid_prim(x):
    return np.exp(-x)/ np.power(1 +np.exp(-x), 2)

N = 1000
x_0 = -10
x_1 = 10
xx = np.linspace(x_0, x_1, N)

sigm = sigmoid(xx)
sigm_prim = sigmoid_prim(xx)

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 6))
ax1.set_xlabel("$x$")
ax1.set_ylabel("$\sigma(x)$")         
ax1.plot(xx, sigm)

ax2.set_xlabel("$x$")
ax2.set_ylabel(r"$\frac{d \sigma}{d x}(x)$")         
ax2.plot(xx, sigm_prim)