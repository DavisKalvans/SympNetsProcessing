import numpy as np
import matplotlib.pyplot as plt 
from pendulum import Pendulum_H


tau = 0.1
tau_txt = str(tau).replace('.', '')

# Number of steps with numerical method
M = 10000
Tend = tau*M
Tend_txt = str(Tend).replace('.0', '')
tm = np.linspace(0, Tend, M+1)
q1 = 0.8
p1 = 0.5

X = np.zeros((2, M+1))
X[:, 0] = [q1, p1]

# Verlet method
for i in range(M):
    p_half = X[1, i] -np.sin(X[0, i])*tau/2
    X[0, i+1] = X[0, i] +p_half*tau
    X[1, i+1] = p_half -np.sin(X[0, i+1])*tau/2


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

# Save results
f = f"Numeric/Pendulum/Verlet_Const{tau_txt}Tau_T{str(int(Tend))}"
np.save(f, X)

# # Check and plot energy
# energy = Pendulum_H(X.T)

# fig2, ax = plt.subplots(figsize=(9, 6.5))
# ax.set_xlabel("$t$")
# ax.set_ylabel("$Energy$")         
# ax.grid(True)
# ax.plot(tm, energy)