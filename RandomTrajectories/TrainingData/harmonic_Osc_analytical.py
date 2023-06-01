import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp   
from harmonic_Osc import HarmOsc

# Time step and other parameters
tau = 0.1
tau_txt = str(tau).replace('.', '')
q1 = 0.3
p1 = 0.5
M = 10000 # Time steps
Tend = tau*M
tm = np.linspace(0, Tend, M+1)

# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

fig1, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xlabel("$q$")
ax.set_ylabel("$p$")         
ax.set_title("Phase portrait")  
ax.grid(True)
ax.axis([-1.6, 1.6, -1, 1])

# Solve with high precision
sol = solve_ivp(HarmOsc, [0, Tend], [q1, p1], method = 'LSODA', t_eval = tm,
                rtol = 1e-12, atol = 1e-12) 

# Plot the trajectory
ax.plot(sol.y[0, :], sol.y[1, :])


# Save results
f = f"Analytical/HarmOsc/Const{tau_txt}Tau_T{str(int(Tend))}"
np.save(f, sol.y)