import numpy as np
from TrainingData.general_problems import Pendulum
from general_training import verletStepNumpy_Pendulum
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch

device = 'cpu'
### Testing
# a = np.array([[2, 4, 4, 4, 5, 5, 7, 9]])
# b = np.array([[1, 8, 2, 3, 4, 8, 1, 5]])
# s_a = np.std(a)
# s_b = np.std(b)
# print(s_a)
# print(s_b)

# ab = np.array([[2, 1], [4, 8], [4, 2], [4, 3], [5, 4], [5, 8], [7, 1], [9, 5]])
# s = np.std(ab, axis=0) # This is what we'll want

# mean = np.mean(ab, axis = 0) # Also what we'll want

x0 = (0.8, 0.5)
extraParams = None
d = 2
Tend = 200
tau = 0.1
M = int(Tend/tau)
tm = np.linspace(0, Tend, M+1)
treshold = 0.05


### Exact solution
ground_truth = solve_ivp(Pendulum.problem, [0, Tend], x0, args = (extraParams,), method = 'RK45', rtol = 1e-12, atol = 1e-12)
exact = solve_ivp(Pendulum.problem, [0, Tend], x0, args = (extraParams,), t_eval = tm, method = 'RK45', rtol = 1e-12, atol = 1e-12)
exact = exact.y.T

### Numerical solution with only core method (Euler)
pred_Euler = np.zeros([M+1, 2])
pred_Euler[0, :] = x0

for i in range(M):
    pred_Euler[i+1, :] = verletStepNumpy_Pendulum(pred_Euler[i, :], tau, extraParams)

### RMNSE calculation for Euler method
square_error = np.square(exact -pred_Euler)
mse = np.mean(square_error, axis=1)
root_mse = np.sqrt(mse)
std = np.std(exact, axis=0)
normalized_square_error = square_error/np.square(std)
mean_normalized_square_error = np.mean(normalized_square_error, axis = 1)
root_mean_normalized_square_error = np.sqrt(mean_normalized_square_error)
# Reduce the wild naming to acronym
rmnse_euler = root_mean_normalized_square_error

### Trained model with the timestep 0.1 and Euler as core method prediction
model, *_ = torch.load('TrainedModels/Verlet/Sym/Pendulum/PendulumRandN40M20Const01Tau50TH_2eta1_2L16n0m')
Z = torch.tensor(x0, dtype=torch.float64, device=device).reshape((1, 1, d))
Tau = torch.tensor([[[tau**2]]], dtype=torch.float64, device=device)

pred_inv = np.zeros([M+1, d])
pred_numeric = np.zeros([M+1, d])
with torch.no_grad():
    inverse, _ = model.back(Z, Tau)
            
pred_inv[0] = inverse.reshape((1, d)).numpy()

for i in range(M):
    pred_inv[i+1] = verletStepNumpy_Pendulum(pred_inv[i], tau, extraParams)

pred_inv = torch.from_numpy(np.float64(pred_inv)).reshape(M+1, 1, d)
with torch.no_grad():
    pred, _ = model(pred_inv, Tau)

pred = pred.reshape((M+1, d)).numpy()

### RMNSE calculation for new method
square_error = np.square(exact -pred)
mse = np.mean(square_error, axis=1)
root_mse = np.sqrt(mse)
std = np.std(exact, axis=0)
normalized_square_error = square_error/np.square(std)
mean_normalized_square_error = np.mean(normalized_square_error, axis = 1)
root_mean_normalized_square_error = np.sqrt(mean_normalized_square_error)
# Reduce the wild naming to acronym
rmnse = root_mean_normalized_square_error

### Calculate the actual VPT

for i in range(M):
    if rmnse_euler[i] > treshold:
        VPT_euler = tau*(i-1)
        break

for i in range(M):
    if rmnse[i] > treshold:
        VPT = tau*(i-1)
        break


### Plot VPT and treshold
fig1, ax = plt.subplots(figsize=(9, 6.5))
ax.plot(tm, np.ones(M+1)*treshold, color = 'k', linewidth = '2', linestyle = '--', label = f'Treshold={treshold}')
ax.plot(tm, rmnse, color = 'g', label = "New method")
ax.plot(tm, rmnse_euler, color = 'r', label = "Euler")
ax.legend(loc=0, prop={'size':20})
ax.grid(True)
ax.set_xlabel('Time')
ax.set_ylabel('RMNSE')
ax.set_title(f"Pendulum with x0={x0}")

print(f'Verlet VPT with timestep {tau} is {VPT_euler}')
print(f'New method VPT with timestep {tau} is {VPT}')
print(f'New method achieves {VPT/VPT_euler} times improvement')

fig2, ax = plt.subplots(figsize=(9, 6.5))
ax.plot(ground_truth.y.T[:, 0], ground_truth.y.T[:, 1], color = 'k', linewidth = '2')
ax.plot(pred[-100::, 0], pred[-100::, 1], color = 'g')
ax.plot(pred_Euler[-100::, 0], pred_Euler[-100::, 1], color = 'r', linestyle = '--')

