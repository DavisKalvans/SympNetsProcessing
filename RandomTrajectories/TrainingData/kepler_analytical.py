import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp   
from kepler import Kepler

# Time step and other parameters
tau = 0.01
tau_txt = str(tau).replace('.', '')
e = 0.6
q1 = 1 - e
q2 = 0
p1 = 0
p2 = np.sqrt((1+e)/(1-e))
M = 700 # Time steps
Tend = tau*M
tm = np.linspace(0, Tend, M+1)

# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

fig1, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xlabel("$q1$")
ax.set_ylabel("$q2$")         
ax.set_title("Phase portrait")  
ax.grid(True)
#ax.axis([-1.6, 1.6, -1, 1])

# Solve with high precision
sol = solve_ivp(Kepler, [0, Tend], [q1, q2, p1, p2], method = 'LSODA', t_eval = tm,
                rtol = 1e-12, atol = 1e-12) 

# Plot the trajectory
ax.plot(sol.y[0, :], sol.y[1, :])


# Save results
#f = f"Analytical/Kepler/Const{tau_txt}Tau_T{str(int(Tend))}"
#np.save(f, sol.y)