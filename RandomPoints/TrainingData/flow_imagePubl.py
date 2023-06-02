import numpy as np
import matplotlib.pyplot as plt 
from general_problems import Pendulum
import verlet8


tau = 0.01
radius = 5
center = 3
D = 1 # Half dimension
M = 20
tau = radius/(M)
x = []
y = []
# Let's create a square, yay!
# Bottom
for i in range(M+1):
    y.append(center)
    x.append(center+i*tau)

for i in range(M+1):
# Right side
    x.append(center+radius)
    y.append(center+i*tau)
# Top 
for i in range(M+1):
    x.append(center +radius-i*tau)
    y.append(center + radius)
# Left side
for i in range(M+1):
    x.append(center)
    y.append(center +radius-i*tau)

# Plotting options
font = {'family' : 'serif',
        'size'   : 16}
plt.rc('font', **font)

fig1, ax = plt.subplots(figsize=(9, 6.5))

'''
plt.tick_params(
    axis='both',       # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
'''

plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)     
ax.grid(True)

ax.plot(x, y, linewidth = 2)

X = np.zeros((2, 4*(M+1)))
X_tranform = np.zeros((2, 4*(M+1)))
for i in range(4*(M+1)):
    X[0, i] = x[i]
    X[1, i] = y[i]
    X_tranform[:, i] = verlet8.eight_orderPrecise(X[:, i], 0.2, D, Pendulum, extraParams=None)

ax.plot(X_tranform[0, :], X_tranform[1, :], linestyle = 'dashed', linewidth = 2)

for i in range(4*(M+1)):
    X_tranform[:, i] = verlet8.eight_orderPrecise(X_tranform[:, i], 0.5, D, Pendulum, extraParams=None)

ax.plot(X_tranform[0, :], X_tranform[1, :], linestyle = 'dashdot', linewidth = 2)
plt.axis('equal')
