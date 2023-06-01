import numpy as np
from TrainingData.general_problems import Pendulum, HarmOsc, Kepler
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
device = 'cpu'


'''
Try to make 4-th order Verlet solver for all of our problems
Great success!
'''
# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

mse_func = torch.nn.MSELoss()
Tend = 10
low_pow = 1
upp_pow = 10
powers = np.arange(low_pow, upp_pow, dtype = np.float64)
taus = 2**(-powers)
d = 2
x0_HarmOsc = (0.3, 0.5)
x0_Pendulum = (0.8, 0.5)
e = 0.6
q1 = 1 - e
q2 = 0
p1 = 0
p2 = np.sqrt((1+e)/(1-e))
x0_Kepler = (q1, q2, p1, p2)
extraParams = None

### Select problem
problem = Kepler
if problem == Pendulum:
    prob_text = "Pendulum"
    x0 = x0_Pendulum
    d = 2

elif problem == HarmOsc:
    prob_text = "HarmOsc"
    x0 = x0_HarmOsc
    extraParams = 0.5
    d = 2

elif problem == Kepler:
    prob_text = "Kepler"
    x0 = x0_Kepler
    d = 4

D = int(d/2)

### Calculate energy of starting point
H0 = problem.H(np.array([x0]).reshape(1, d), extraParams)

# Inbuilt solve to compare if I haven't messed up
solver = solve_ivp(problem.problem, [0, Tend], x0, args = (extraParams,), method = 'RK45', rtol = 1e-12, atol = 1e-12)
solver = solver.y.T[-1] # Only want the solution at time t=Tend
solver = np.array(solver)
solver = torch.tensor(solver, dtype = torch.float64, device = device).reshape((1, d))




# Triple jump coefficients (Hairier 4.4)
def gen_gamma(p):
    gamma = np.zeros(3)
    gamma[0] = 1/(2-2**(1/(p+1)))
    gamma[1] = -(2**(1/(p+1))) /(2-2**(1/(p+1)))
    gamma[2] = 1/(2-2**(1/(p+1)))

    return gamma

def fourth_order(x0, tau, gamma, problem, extraParams):
    tau_half = tau/2

    # First time
    p = x0[D:(2*D)] - gamma[0]*tau_half*problem.H_p(x0[0:D], extraParams)
    Q = x0[0:D] +gamma[0]*tau*problem.H_q(p, extraParams)
    P = p - gamma[0]*tau_half*problem.H_p(Q, extraParams)

    # Second time
    p = P - gamma[1]*tau_half*problem.H_p(Q, extraParams)
    Q = Q + gamma[1]*tau*problem.H_q(p, extraParams)
    P = p - gamma[1]*tau_half*problem.H_p(Q, extraParams)

    # Third time
    p = P - gamma[2]*tau_half*problem.H_p(Q, extraParams)
    Q = Q + gamma[2]*tau*problem.H_q(p, extraParams)
    P = p - gamma[2]*tau_half*problem.H_p(Q, extraParams)

    return np.concatenate((Q, P))

def sixth_order(x0, tau, gamma2, gamma4, problem, extraParams):
    # First time
    tau1 = tau*gamma4[0]
    x1 = fourth_order(x0, tau1, gamma2, problem, extraParams)

    #Second time
    tau2 = tau*gamma4[1]
    x2 = fourth_order(x1, tau2, gamma2, problem, extraParams)

    # Third time
    tau3 = tau*gamma4[2]
    x3 = fourth_order(x2, tau3, gamma2, problem, extraParams)

    return x3

def eight_order(x0, tau, gamma2, gamma4, gamma6, problem, extraParams):
    # First time
    tau1 = tau*gamma6[0]
    x1 = sixth_order(x0, tau1, gamma2, gamma4, problem, extraParams)

    tau2 = tau*gamma6[1]
    x2 = sixth_order(x1, tau2, gamma2, gamma4, problem, extraParams)

    tau3 = tau*gamma6[2]
    x3 = sixth_order(x2, tau3, gamma2, gamma4, problem, extraParams)

    return x3


gamma2 = gen_gamma(2)
gamma4 = gen_gamma(4)
gamma6 = gen_gamma(6)
# Sanity check, if the coefficients are correct
#sum = gamma.sum() # Should be one
#sum_powers = gamma[0]**(p+1) +gamma[1]**(p+1) +gamma[2]**(p+1) # Should be zero
#print(sum)
#print(sum_powers)

pred = []
energies = []
energies_solver = []

### Get convergence by using each time step in taus using the eight order numerical method
for tau in taus:
    M = int(Tend/tau) # Time steps
    tm = np.linspace(0, Tend, M+1)

    pred_solver = solve_ivp(problem.problem, [0, Tend], x0, args = (extraParams,), t_eval = tm, method = 'RK45', rtol = 1e-12, atol = 1e-12)
    pred_solver = pred_solver.y.T
    pred_solver = torch.tensor(pred_solver, dtype = torch.float64, device = device).reshape((M+1, d))
    energies_solver.append(problem.H(pred_solver, extraParams))

    pred_numeric = np.zeros([M+1, d])
    pred_numeric[0, :] = x0

    for i in range(M):
        ### Function eight order
        pred_numeric[i+1, :] = eight_order(pred_numeric[i, :], tau, gamma2, gamma4, gamma6, problem, extraParams)


    pred.append(torch.tensor(pred_numeric[-1]))
    energies.append(problem.H(pred_numeric, extraParams))


### Calculate errors vs solve_ivp
errors = np.zeros(len(taus))
errors_energy = np.zeros(len(taus))
errors_energy_solver = np.zeros(len(taus))

for i in range(len(taus)):
    errors[i] = mse_func(pred[i], solver)
    errors_energy[i] = max(abs(energies[i] -H0))
    errors_energy_solver[i] = max(abs(energies_solver[i] -H0))

### Plot everything
line1 = np.array(taus)**8

fig1, ax = plt.subplots(figsize=(9, 6.5))
ax.loglog(np.array(taus), errors**(0.5), color = 'r', linewidth = '2', label = 'Verlet')
ax.loglog(np.array(taus), line1, color = 'k', label = "eight order")
ax.legend(loc=4, prop={'size':20})
ax.grid(True)
ax.set_xlabel(r'$\tau$')
ax.set_ylabel('Aboslute error')
ax.set_title(f"{prob_text} with x0={x0}")

fig2, ax = plt.subplots(figsize=(9, 6.5))
ax.loglog(np.array(taus), errors_energy, color = 'r', linewidth = '2', label = 'Verlet')
ax.loglog(np.array(taus), errors_energy_solver, color = 'g', linewidth = '2', label = 'solve_ivp RK45')
ax.loglog(np.array(taus), line1, color = 'k', label = "eight order")
ax.legend(loc=4, prop={'size':20})
ax.grid(True)
ax.set_xlabel(r'$\tau$')
ax.set_ylabel('Max Hamiltonian error')
ax.set_title(f"Pendulum with x0={x0}")