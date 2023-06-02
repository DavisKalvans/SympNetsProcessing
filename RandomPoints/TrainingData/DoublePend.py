import numpy as np

def DoublePendulum(t, z):
    g = 1

    
    q1, q2, p1, p2 = z
    h1 = p1*p2*np.sin(q1-q2)
    h1 /= 1 +np.sin(q1-q2)**2
    h2 = p1**2 +2*p2**2 -2*p1*p2*np.cos(q1-q2)
    h2 /= 2 *(1 +np.sin(q1-q2)**2)**2
    f = np.zeros(4)
    f[0] = p1 -p2*np.cos(q1-q2)
    f[0] /= 1 +np.sin(q1-q2)**2
    f[1] = -p1*np.cos(q1-q2) +2*p2
    f[1] /= 1 +np.sin(q1-q2)**2
    f[2] = -2*g*np.sin(q1) -h1 +h2*np.sin(2*q1 -2*q2)
    f[3] = -g*np.sin(q2) +h1 -h2*np.sin(2*q1 -2*q2)
    
    return f    

# Hamiltonian of Double Pendulum problem
def DoublePendulum_H(z):
    g = 1
    
    q1 = z[:, 0]
    q2 = z[:, 1]
    p1 = z[:, 2]
    p2 = z[:, 3]
    
    H = p1**2 +2*p2**2 -2*p1*p2*np.cos(q1-q2)
    H /= 2 +2*np.sin(q1-q2)**2
    H -= 2*g*np.cos(q1) + g*np.cos(q2)
    return H