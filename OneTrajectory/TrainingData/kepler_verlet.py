import numpy as np
import matplotlib.pyplot as plt 
from kepler import Kepler_H, Kepler_L


tau = 0.1
tau_txt = str(tau).replace('.', '')

# Number of steps with numerical method
M = 10000
Tend = tau*M
Tend_txt = str(Tend).replace('.0', '')
tm = np.linspace(0, Tend, M+1)

e = 0.6
q1 = 1 - e
q2 = 0
p1 = 0
p2 = np.sqrt((1+e)/(1-e))

X = np.zeros((4, M+1))
X[:, 0] = [q1, q2, p1, p2]

# Verlet method
for i in range(M):
    q1_half = X[0, i] +X[2, i]*tau/2
    q2_half = X[1, i] +X[3, i]*tau/2

    d = (q1_half**2 +q2_half**2)**(1.5)
    X[2, i+1] = X[2, i] -q1_half/d*tau
    X[3, i+1] = X[3, i] -q2_half/d*tau

    X[0, i+1] = q1_half +X[2, i+1]*tau/2
    X[1, i+1] = q2_half +X[3, i+1]*tau/2



# Plotting options
font = {'family' : 'serif',
        'size'   : 16}
plt.rc('font', **font)

fig1, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xlabel("$q1$")
ax.set_ylabel("$q2$")         
ax.grid(True)

# Plot the trajectory
ax.plot(X[0, :], X[1, :])

# Save results
f = f"Numeric/Kepler/Verlet_Const{tau_txt}Tau_T{str(int(Tend))}"
np.save(f, X)

# Check and plot energy
energy = Kepler_H(X.T)

fig2, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xlabel("$t$")
ax.set_ylabel("$Energy$")         
ax.grid(True)
ax.plot(tm, energy)

# Check and plot angular mommentum
angMoment = Kepler_L(X.T)

fig3, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xlabel("$t$")
ax.set_ylabel("$Angular momentum$")         
ax.grid(True)
ax.plot(tm, angMoment)