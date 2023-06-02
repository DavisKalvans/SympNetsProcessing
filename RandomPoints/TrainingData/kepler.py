import numpy as np

# Kepler problem
def Kepler(t, z):
    q1, q2, p1, p2 = z
    d = (q1**2 + q2**2)**(1.5)
    f = np.zeros(4)
    f[0] = p1
    f[1] = p2
    f[2] = -q1/d
    f[3] = -q2/d
    return f

# Hamiltonian of the Kepler problem
def Kepler_H(z):
    q1 = z[:, 0]
    q2 = z[:, 1]
    p1 = z[:, 2]
    p2 = z[:, 3]
    H = (p1**2 + p2**2)/2
    H -= 1.0/np.sqrt(q1**2 + q2**2)
    return H

# Angular momentum of the Kepler problem
def Kepler_L(z):
    q1 = z[:, 0]
    q2 = z[:, 1]
    p1 = z[:, 2]
    p2 = z[:, 3]
    L = q1*p2 - q2*p1
    return L