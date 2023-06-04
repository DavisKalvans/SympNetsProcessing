import numpy as np
### Classes with problems that contain all necessary functions.
### .problem for solve_ivp, .H for calculating Hamiltonian energy
### .H_q and .H_p for verlet8.eightOrderPrecise


class HarmOsc():
    def problem(t, y, omega): # Problem for odesolvers 
        q, p = y
        f = np.zeros(2)
        f[0] = p
        f[1] = -omega**2*q
        return f
    
    def H(z, omega): # Hamiltonian
        q = z[:, 0]
        p = z[:, 1]
        H = p**2/2 + omega**2*(q**2)/2
        return H
    
    def H_q(x, omega):
        return x
    
    def H_p(x, omega):
        return omega**2*x

class Pendulum():

    def problem(t, z, extraParams = None): # Problem for odesolvers
        q, p = z
        f = np.zeros(2)
        f[0] = p
        f[1] = -np.sin(q)
        return f 
    
    def H(z, extraParams = None): # Hamiltonian
        q = z[:, 0]
        p = z[:, 1]
        H = p**2/2 - np.cos(q)
        return H
    
    def H_q(x, extraParams = None):
        return x
    
    def H_p(x, extraParams = None):
        return np.sin(x)

# Not used atm, mitgh not be totally correct too
class Lennard_Jones():
    def V_prim(r):
        a1 = -12*(0.341/r)**12/r
        a2 = 6*(0.341//r)**6/r
        a = a1+a2
        a = a*4*119.8
        return a

    def H_q(x, extraParams = None):
        f = np.zeros(4)
        f[0] = x[0]
        f[1] = x[1]
        f[2] = x[2]
        f[3] = x[3]
        return f
    
    def H_p(x, extraParams = None):
        f = np.zeros(4)
        d1 = x[0] -x[1]
        d2 = x[3] -x[0]
        f[0] = 4*119.8*(-12*(0.341/d1)**12/d1 +6*(0.341/d2)**6/d1 +12*(0.341/d2)**12/d2 -6*(0.341/d2)**6/d2)
        d1 = x[1] -x[2]
        d2 = x[0] -x[1]
        f[1] = 4*119.8*(-12*(0.341/d1)**12/d1 +6*(0.341/d2)**6/d1 +12*(0.341/d2)**12/d2 -6*(0.341/d2)**6/d2)
        d1 = x[2] -x[3]
        d2 = x[1] -x[2]
        f[2] = 4*119.8*(-12*(0.341/d1)**12/d1 +6*(0.341/d2)**6/d1 +12*(0.341/d2)**12/d2 -6*(0.341/d2)**6/d2)
        d1 = x[3] -x[0]
        d2 = x[2] -x[3]
        f[3] = 4*119.8*(-12*(0.341/d1)**12/d1 +6*(0.341/d2)**6/d1 +12*(0.341/d2)**12/d2 -6*(0.341/d2)**6/d2)
        
        
        #f[0] = self.V_prim(x[0] -x[1]) -self.V_prim(x[3] -x[0])
        #f[1] = self.V_prim(x[1] -x[2]) -self.V_prim(x[0] -x[1])
        #f[2] = self.V_prim(x[2] -x[3]) -self.V_prim(x[1] -x[2])
        #f[3] = self.V_prim(x[3] -x[0]) -self.V_prim(x[2] -x[3])
        return f

    def V(self, r):
        a = np.power(0.341/r, 12) -np.power(0.341/r, 6)
        return 4*119.8*a
    
    
    def H(self, z, extraParams = None):
        V_vec = np.vectorize(self.V)
        q1 = z[:, 0]
        q2 = z[:, 1]
        q3 = z[:, 2]
        q4 = z[:, 3]
        p1 = z[:, 4]
        p2 = z[:, 5]
        p3 = z[:, 6]
        p4 = z[:, 7]
        #return q1-q2
        H = (p1**2 +p2**2 +p3**2 +p4**2)/2
        H += self.V(q1-q2)
        H += self.V(q2-q3)
        H += self.V(q3-q4)
        H += self.V(q4-q1)
        return H

class Kepler():

    def problem(t, z, extraParams = None): # Problem for odesolvers
        q1, q2, p1, p2 = z
        d = (q1**2 + q2**2)**(1.5)
        f = np.zeros(4)
        f[0] = p1
        f[1] = p2
        f[2] = -q1/d
        f[3] = -q2/d
        return f
    
    def H(z, extraParams = None): # Hamiltonian
        q1 = z[:, 0]
        q2 = z[:, 1]
        p1 = z[:, 2]
        p2 = z[:, 3]
        H = (p1**2 + p2**2)/2
        H -= 1.0/np.sqrt(q1**2 + q2**2)
        return H
    
    def L(z, extraParams = None): # Angular momentum
        q1 = z[:, 0]
        q2 = z[:, 1]
        p1 = z[:, 2]
        p2 = z[:, 3]
        L = q1*p2 - q2*p1
        return L
    
    def H_q(x, extraParams = None):
        f = np.zeros(2)
        f[0] = x[0]
        f[1] = x[1]
        return f

    def H_p(x, extraParams = None):
        f = np.zeros(2)
        d = (x[0]**2 +x[1]**2)**(1.5)
        f[0] = x[0]/d
        f[1] = x[1]/d
        return f
    
# Not a separable Hamiltonian system, would need to use implicit solvers
class DoublePendulum():
    def problem(t, z):
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
    
    def H(z, extraParams = None):
        g = 1
    
        q1 = z[:, 0]
        q2 = z[:, 1]
        p1 = z[:, 2]
        p2 = z[:, 3]
        
        H = p1**2 +2*p2**2 -2*p1*p2*np.cos(q1-q2)
        H /= 2 +2*np.sin(q1-q2)**2
        H -= 2*g*np.cos(q1) + g*np.cos(q2)
        return H
    
