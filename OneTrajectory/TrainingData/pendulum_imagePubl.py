import numpy as np
import matplotlib.pyplot as plt 
from general_problems import Pendulum
import verlet8


tau = 0.05
tau_txt = str(tau).replace('.', '')

# Number of steps with numerical method
M = 1000
Tend = tau*M
Tend_txt = str(Tend).replace('.0', '')
tm = np.linspace(0, Tend, M+1)
D = 1 # Half dimension
q1 = 2.5
p1 = 0.5

X = np.zeros((2, M+1))
X[:, 0] = [q1, p1]
X2 = np.zeros((2, M+1))
X2[:, 0] = [0.8, 0.3]
X3 = np.zeros((2, M+1))
X3[:, 0] = [1.7, -0.2]

# Verlet method
for i in range(M):
    X[:, i+1] = verlet8.eight_orderPrecise(X[:, i], tau, D, Pendulum, extraParams=None)
    X2[:, i+1] = verlet8.eight_orderPrecise(X2[:, i], tau, D, Pendulum, extraParams=None)
    X3[:, i+1] = verlet8.eight_orderPrecise(X3[:, i], tau, D, Pendulum, extraParams=None)


# Plotting options
font = {'family' : 'serif',
        'size'   : 16}
plt.rc('font', **font)

fig1, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xlabel("$q$")
ax.set_ylabel("$p$")         
ax.grid(True)

# Plot the trajectory
ax.plot(X[0, :], X[1, :])
ax.plot(X[0, 0], X[1, 0], 'ro')
ax.plot(X3[0, :], X3[1, :])
ax.plot(X3[0, 0], X3[1, 0], 'ro')
ax.plot(X2[0, :], X2[1, :])
ax.plot(X2[0, 0], X2[1, 0], 'ro')


# Check and plot energy
energy = Pendulum.H(X.T)
energy2 = Pendulum.H(X2.T)
energy3 = Pendulum.H(X3.T)

fig2, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xlabel("$t$")
ax.set_ylabel("$H(q, p)$")         
ax.grid(True)
ax.plot(tm, energy, linewidth=3)
ax.plot(tm, energy3, linewidth=3)
ax.plot(tm, energy2, linewidth=3)