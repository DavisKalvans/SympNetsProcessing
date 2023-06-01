import numpy as np
from general_problems import Lennard_Jones
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp   

### NOT USED HERE
#EPSILON = 119.8
#SIGMA = 0.341
#MASS = 66.34*10**(-27)

#p = np.array([[0, 0], [0.02, 0.39], [0.34, 0.17], [0.36, -0.21], [-0.02, -0.4], [-0.35, -0.16], [-0.31, 0.21]])
#q = np.array([[-30, -20], [50, -90], [-70, -60], [90, 40], [80, 90], [-40, 100], [-80, -60]])
p = [-1.5, 1, 0.5, 1.5]
q = [1, 2, -1, -2]
N = len(q)

### NOT USED
#def V(r):
#    a = (SIGMA/r)**12 -(SIGMA/r)**6
#    return 4*EPSILON*a

#def H(q, p, N):
#    sum1 = 0
#    for i in range(N):
#        sum1 += p[i]**2
#
#    sum1 /=2
#
#    sum2 = 0
#    for i in range(N-1):
#        sum2 += V(q[i]-q[i+1])
#
#    sum2 += V(q[N-1]-q[0])
#
#    sum = sum1+sum2
#    return sum

#x = [1, 2, -1, -2, -1.5, 1, 0.5, 1.5]
#x0 = np.zeros([1, 8])
#x0[0, :] =x
#x = np.array(x)
#x.reshape((1, 8))
#H1 = H(q, p, N)
#H2 = lj.H(x0)

D = 4 # Half dimension
tau = 10**(-10) # Timestep
probl_class = Lennard_Jones

#x0 = (1, 2, -1, -2, -1.5, 1, 0.5, 1.5)
x0 = ( 1, 2, 3, 4, -0.4, 0.45, 0.37, -0.48)
M = 100000
exact = np.zeros([M+1, 8])
exact[0] = x0
for i in range(M):
    exacttt = solve_ivp(Lennard_Jones.problem, [0, tau], exact[i], method = 'RK45',
                rtol = 1e-12, atol = 1e-12) 
    exact[i+1] = exacttt.y.T[-1].reshape((1, 8))

lj = Lennard_Jones()
H2 = lj.H(exact)
plt.figure(1)
plt.plot(exact[:, 0], exact[:, 4])
plt.xlabel('q1')
plt.ylabel('p1')
plt.figure(2)
plt.plot(exact[:, 1], exact[:, 5])
plt.xlabel('q2')
plt.ylabel('p2')
plt.figure(3)
plt.plot(exact[:, 2], exact[:, 6])
plt.xlabel('q3')
plt.ylabel('p3')
plt.figure(4)
plt.plot(exact[:, 3], exact[:, 7])
plt.xlabel('q4')
plt.ylabel('p4')


plt.figure(5)
Tend = tau*M
tm = np.linspace(0, Tend, M+1)
plt.plot(tm, H2)
plt.title("Hamiltonian energy")

