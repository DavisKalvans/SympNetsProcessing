import numpy as np

# Harmonic oscillator equation
def HarmOsc(t, z):
    omega = 0.5
    q, p = z
    f = np.zeros(2)
    f[0] = p
    f[1] = -omega**2*q
    return f

# Harmonic oscillator Hamiltonian
def HarmOsc_H(z):
    omega = 0.5
    q = z[:, 0]
    p = z[:, 1]
    H = p**2/2 + (omega**2)*(q**2)/2
    return H

