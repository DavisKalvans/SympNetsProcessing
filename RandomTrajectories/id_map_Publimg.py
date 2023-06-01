import numpy as np
import torch
from general_training import eulerStepNumpy_Pendulum, verletStepNumpy_Pendulum
import matplotlib.pyplot as plt

# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

# Paramters
x0_Pendulum = (0.8, 0.5)
powers = np.arange(2, 14, dtype=np.float64)
taus = np.power(2, -powers)


### Euler kernel and non-sym
# Load model and the kernel
numeric_stepNumpy = eulerStepNumpy_Pendulum
model_name = "TrainedModels/Euler/Pendulum/PendulumRandN20M10Const01Tau50TH_2eta1_2L8n0m"
model, *_ = torch.load(model_name)

# Convert to tensors for model
device = 'cpu'
d = len(x0_Pendulum)
Z = torch.tensor(x0_Pendulum, dtype=torch.float64, device=device).reshape((1, 1, d))
inverses = []
for tau in taus:
    Tau = torch.tensor([[[tau]]], dtype=torch.float64, device = device)

    with torch.no_grad():
        inverse, _ = model.back(Z, Tau)

    inverses.append(inverse)

norms = []
for inverse in inverses:
    inverse = inverse.reshape(2).numpy()
    norms.append(np.sqrt(np.sum((inverse-x0_Pendulum)**2, 0)))

fig, ax = plt.subplots(figsize=(9, 6.5))
ax.loglog(taus, norms, label=r"$\|\Psi(x)-x\|$", linewidth = '2', ls = '--')
line1 = np.array(taus)
ax.loglog(taus, line1, label="Pirmā kārta", linewidth = '2')
ax.legend()

### Kepler kernel and non-sym
# Load model and the kernel
numeric_stepNumpy = eulerStepNumpy_Pendulum
model_name = "TrainedModels/Verlet/Sym/Pendulum/PendulumRandN80M100Const01Tau50TH_2eta1_2L8n0m"
model, *_ = torch.load(model_name)

# Convert to tensors for model
device = 'cpu'
d = len(x0_Pendulum)
Z = torch.tensor(x0_Pendulum, dtype=torch.float64, device=device).reshape((1, 1, d))
inverses = []
for tau in taus:
    Tau = torch.tensor([[[tau**2]]], dtype=torch.float64, device = device)

    with torch.no_grad():
        inverse, _ = model.back(Z, Tau)

    inverses.append(inverse)

norms = []
for inverse in inverses:
    inverse = inverse.reshape(2).numpy()
    norms.append(np.sqrt(np.sum((inverse-x0_Pendulum)**2, 0)))

fig, ax = plt.subplots(figsize=(9, 6.5))
ax.loglog(taus, norms, label=r"$\|\Psi(x)-x\|$", linewidth = '2', ls = '--')
line2 = np.array(taus)**2
ax.loglog(taus, line2, label="Otrā kārta", linewidth = '2')
ax.legend()
