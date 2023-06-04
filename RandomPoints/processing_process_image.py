import numpy as np
import torch
from general_training import eulerStepNumpy_Pendulum
import matplotlib.pyplot as plt

# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

# Load model and the kernel
numeric_stepNumpy = eulerStepNumpy_Pendulum
model_name = "TrainedModels/Euler/Pendulum/PendulumRandN20M10Const01Tau50TH_2eta1_2L8n0m"
model, *_ = torch.load(model_name)

# Paramters
x0_Pendulum = (0.8, 0.5)
tau = 0.1
M = 5 # Nr of steps

# Convert to tensors for model
device = 'cpu'
d = len(x0_Pendulum)
Z = torch.tensor(x0_Pendulum, dtype=torch.float64, device=device).reshape((1, 1, d))
Tau = torch.tensor([[[tau]]], dtype=torch.float64, device = device)

# Do the predictions
pred_inv = np.zeros([M+1, d])
with torch.no_grad():
    inverse, _ = model.back(Z, Tau)

pred_inv[0] = inverse.reshape((1, d)).numpy()

for i in range(M):
    pred_inv[i+1] = numeric_stepNumpy(pred_inv[i], tau, None)

pred_inv = torch.from_numpy(pred_inv).reshape((M+1, 1, d))
with torch.no_grad():
    pred, _ = model(pred_inv, Tau)

pred = pred.numpy().reshape(M+1, d)

# Plotting
fig, ax = plt.subplots(figsize=(9, 6.5))
pred_inv = pred_inv.reshape((M+1, d)).numpy()
# Points in special space
ax.plot(pred_inv[:, 0], pred_inv[:, 1], color = 'r', ls='--', marker = '*', label = r'$K_h$')
# Predictions or points in original space
ax.plot(pred[:, 0], pred[:, 1], color = 'b', marker = 'o', label = r'$\hat{K}_h$')
ax.plot(pred[0, 0], pred[0, 1], color = 'b', marker = 'o', ms='12')
ax.legend()
#ax.grid(True)
ax.set_xlabel("$q$")
ax.set_ylabel("$p$")         

# Calculate everything for arrows
dq = []
dp = []

for i in range(1, M+1):
    dq.append(-pred_inv[i, 0] +pred[i, 0])
    dp.append(-pred_inv[i, 1] +pred[i, 1])

for i in range(M):
    ax.arrow(pred_inv[i+1, 0], pred_inv[i+1, 1], dq[i], dp[i], color = 'k', length_includes_head=True, aa = True, overhang = 0, linewidth = 1, head_length =0.01)

ax.arrow(pred[0, 0], pred[0, 1], pred_inv[0, 0]-pred[0, 0], pred_inv[0, 1]-pred[0, 1], color = 'k', length_includes_head=True, aa = True, overhang = 0, linewidth = 1, head_length =0.01)