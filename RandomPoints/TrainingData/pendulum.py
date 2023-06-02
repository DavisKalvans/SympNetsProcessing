import numpy as np

#  Mathematical pendulum equation
def Pendulum(t, z):
    q, p = z
    f = np.zeros(2)
    f[0] = p
    f[1] = -np.sin(q)
    return f

# Mathematical pendulum Hamiltonian
def Pendulum_H(z):
    q = z[:, 0]
    p = z[:, 1]
    H = p**2/2 - np.cos(q)
    return H
