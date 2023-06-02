import numpy as np
from TrainingData.general_problems import Pendulum
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
device = 'cpu'


'''
Compares float32 and float64 accuracy when performing the second order Verlet method on the
Pendulum problem using PyTorch against NumPy.
'''
# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

mse_func = torch.nn.MSELoss()
Tend = 1
low_pow = 1
upp_pow = 16
powers = np.arange(low_pow, upp_pow, dtype = np.float64)
taus = 2**(-powers)
d = 2

x0 = (0.8, 0.5)

exact = solve_ivp(Pendulum.problem, [0, Tend], x0, args = (None,), method = 'RK45', rtol = 1e-12, atol = 1e-12)
exact = exact.y.T[-1] # Only want the solution at time t=Tend
exact = np.array(exact)
exact = torch.tensor(exact, dtype = torch.float64, device = device).reshape((1, d))

H0 = Pendulum.H(np.array([x0]).reshape(1, d))

numeric = []
energies = []

numeric_torch = []
energies_torch = []
numeric_torch32 = []
energies_torch32 = []

for tau in taus:
    M = int(Tend/tau) # Time steps
    tm = np.linspace(0, Tend, M)

    pred_numeric = np.zeros([M+1, d])
    pred_torch = torch.zeros([M+1, d], dtype = torch.float64)
    pred_torch32 = torch.zeros([M+1, d], dtype = torch.float32)

    pred_numeric[0, :] = x0
    pred_torch[0, :] = torch.tensor(x0, dtype = torch.float64)
    pred_torch32[0, :] = torch.tensor(x0, dtype = torch.float32)
    tau_half = tau/2

    tau_torch = torch.tensor(tau, dtype = torch.float64)
    tau_half_torch = tau_torch/2

    tau_torch32 = torch.tensor(tau, dtype = torch.float32)
    tau_half_torch32 = tau_torch/2

    for i in range(M):
        p_half = pred_numeric[i, 1] -tau_half*np.sin(pred_numeric[i, 0])
        pred_numeric[i+1, 0] = pred_numeric[i, 0] +tau*p_half
        pred_numeric[i+1, 1] = p_half -tau_half*np.sin(pred_numeric[i+1, 0])

        pp_half = pred_torch[i, 1] -tau_half_torch*torch.sin(pred_torch[i, 0])
        pred_torch[i+1, 0] = pred_torch[i, 0] +tau_torch*pp_half
        pred_torch[i+1, 1] = pp_half -tau_half_torch*torch.sin(pred_torch[i+1, 0])

        ppp_half = pred_torch32[i, 1] -tau_half_torch32*torch.sin(pred_torch32[i, 0])
        pred_torch32[i+1, 0] = pred_torch32[i, 0] +tau_torch32*ppp_half
        pred_torch32[i+1, 1] = ppp_half -tau_half_torch32*torch.sin(pred_torch32[i+1, 0])


    numeric.append(torch.tensor(pred_numeric[-1]))
    energies.append(Pendulum.H(pred_numeric, None))

    numeric_torch.append(pred_torch[-1])
    energies_torch.append(Pendulum.H(pred_torch, None))
    numeric_torch32.append(pred_torch32[-1])
    energies_torch32.append(Pendulum.H(pred_torch32, None))


errors = np.zeros(len(taus))
errors_energy = np.zeros(len(taus))
errors_torch = np.zeros(len(taus))
errors_energy_torch = np.zeros(len(taus))
errors_torch32 = np.zeros(len(taus))
errors_energy_torch32 = np.zeros(len(taus))

for i in range(len(taus)):
    errors[i] = mse_func(numeric[i], exact)
    errors_energy[i] = max(abs(energies[i] -H0))

    errors_torch[i] = mse_func(numeric_torch[i], exact)
    errors_energy_torch[i] = max(abs(energies_torch[i] -H0))

    errors_torch32[i] = mse_func(numeric_torch32[i], exact)
    errors_energy_torch32[i] = max(abs(energies_torch32[i] -H0))

line1 = np.array(taus)**2


fig1, ax = plt.subplots(figsize=(9, 6.5))
#ax.loglog(np.array(taus), errors**(0.5), color = 'r', linewidth = '3', label = 'Verlet Numpy')
ax.loglog(np.array(taus), errors_torch**(0.5), color = 'g', linewidth = '3', label = '64 bitu precizitāte')
ax.loglog(np.array(taus), errors_torch32**(0.5), linestyle = 'dashed', color = 'r', linewidth = '2', label = '32 bitu precizitāte')
#ax.loglog(np.array(taus), line1, color = 'k', label = "second order")
ax.legend(loc=4, prop={'size':20})
ax.grid(True)
ax.set_xlabel(r'$h$')
ax.set_ylabel(r'Aboslūtā kļūda laikā T')
#ax.set_title(f"Pendulum with x0={x0}")

fig2, ax = plt.subplots(figsize=(9, 6.5))
#ax.loglog(np.array(taus), errors_energy, color = 'r', linewidth = '3', label = 'Verlet Numpy')
ax.loglog(np.array(taus), errors_energy_torch, color = 'g', linewidth = '3', label = '64 bitu precizitāte')
ax.loglog(np.array(taus), errors_energy_torch32, linestyle = 'dashed', color = 'r', linewidth = '2', label = '32 bitu precizitāte')
#ax.loglog(np.array(taus), line1, color = 'k', label = "second order")
ax.legend(loc=4, prop={'size':20})
ax.grid(True)
ax.set_xlabel(r'$h$')
ax.set_ylabel(r'$\max{|H(t)-H_0|}$')
#ax.set_title(f"Pendulum with x0={x0}")




