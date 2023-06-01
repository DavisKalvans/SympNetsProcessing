import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from  TrainingData.general_problems import Pendulum
from general_training import eulerStepNumpy_Pendulum

torch.set_num_threads(1)
device = 'cpu'

# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

torch.set_default_dtype(torch.float64) # Yess, more default precision, even though I already set it up manually

x0_Pendulum = (0.8, 0.5)
x0 = x0_Pendulum
d = len(x0)
extraParams = None
Tend = 10
tau = 0.1
M = int(Tend/tau) # Time steps
tm = np.linspace(0, Tend, M+1)

### Solver to compare with
exact = solve_ivp(Pendulum.problem, [0, Tend], x0, args = (extraParams,), method = 'RK45', t_eval=tm, rtol = 1e-12, atol = 1e-12)
exact = exact.y.T

### Regular Euler
pred_Euler = np.zeros([M+1, 2])
pred_Euler[0, :] = x0_Pendulum

for i in range(M):
    pred_Euler[i+1, :] = eulerStepNumpy_Pendulum(pred_Euler[i, :], tau, extraParams)


### Trained model with the timestep 0.1
model, *_ = torch.load('TrainedModels/Euler/Pendulum/PendulumRandN10M5Const01Tau50TH_2eta1_2L16n0m')

Z = torch.tensor(x0, dtype=torch.float64, device=device).reshape((1, 1, d))
Tau = torch.tensor([[[tau]]], dtype=torch.float64, device=device)
pred_inv = np.zeros([M+1, d])
pred_numeric = np.zeros([M+1, d])
with torch.no_grad():
    inverse, _ = model.back(Z, Tau)
            
pred_inv[0] = inverse.reshape((1, d)).numpy()

for i in range(M):
    pred_inv[i+1] = eulerStepNumpy_Pendulum(pred_inv[i], tau, extraParams)

pred_inv = torch.from_numpy(np.float64(pred_inv)).reshape(M+1, 1, d)
with torch.no_grad():
    pred, _ = model(pred_inv, Tau)

pred_model = pred.reshape((M+1, d))


### Magic with triple method higher order
### Get one point trough, then do it the second time with thhe resulting point,
### And now with the resluting point do it the third time, but this time get
### Euluer for M times

### Appears not to work
# Get coefficients
# p=2
# gamma = np.zeros(3)
# gamma[0] = 1/(2-2**(1/(p+1)))
# gamma[1] = -(2**(1/(p+1))) /(2-2**(1/(p+1)))
# gamma[2] = 1/(2-2**(1/(p+1)))

# pred_triple_inv = np.zeros([M+1, d])

# # First time
# Z = torch.tensor(x0, dtype=torch.float64, device=device).reshape((1, 1, d))
# Tau = torch.tensor([[[tau]]], dtype=torch.float64, device=device)
# with torch.no_grad():
#     inverse, _ = model.back(Z, Tau*gamma[0])
            
# pred1 = inverse.reshape((d)).numpy()
# pred2 = np.array(eulerStepNumpy_Pendulum(pred1, tau*gamma[0], extraParams))
# pred2 = torch.from_numpy(pred2).reshape(1, 1, d)
# with torch.no_grad():
#     pred3, _ = model(pred2, Tau*gamma[0])

# # Second time
# Z = pred3.clone().detach()
# with torch.no_grad():
#     inverse, _ = model.back(Z, Tau*gamma[1])
            
# pred1 = inverse.reshape((d)).numpy()
# pred2 = np.array(eulerStepNumpy_Pendulum(pred1, tau*gamma[1], extraParams))
# pred2 = torch.from_numpy(pred2).reshape(1, 1, d)
# with torch.no_grad():
#     pred3, _ = model(pred2, Tau*gamma[1])

# # Third time
# Z = pred3.clone().detach()
# with torch.no_grad():
#     inverse, _ = model.back(Z, Tau*gamma[2])
            
# pred1 = inverse.reshape((d)).numpy()
# pred_triple_inv[0, :] = pred1
# for i in range(M):
#     pred_triple_inv[i+1, :] = np.array(eulerStepNumpy_Pendulum(pred_triple_inv[i, :], tau*gamma[2], extraParams))
# pred_triple_inv = torch.from_numpy(pred_triple_inv).reshape(M+1, 1, d)
# with torch.no_grad():
#     pred_triple, _ = model(pred_triple_inv, Tau*gamma[2])

# pred_triple = pred_triple.reshape((M+1, d))

### Magical triple higher order
### Transport with orignial tau, do three eulers with differing taus
### Transporrt back with original tau

### Also doesn't work

# Get coefficients
p=2
gamma = np.zeros(3)
gamma[0] = 1/(2-2**(1/(p+1)))
gamma[1] = -(2**(1/(p+1))) /(2-2**(1/(p+1)))
gamma[2] = 1/(2-2**(1/(p+1)))

Z = torch.tensor(x0, dtype=torch.float64, device=device).reshape((1, 1, d))
Tau = torch.tensor([[[tau]]], dtype=torch.float64, device=device)
pred_inv = np.zeros([M+1, d])
pred_numeric = np.zeros([M+1, d])
with torch.no_grad():
    inverse, _ = model.back(Z, Tau)
            
# First time
pred_inv[0] = inverse.reshape((1, d)).numpy()
for i in range(M):
    pred_inv[i+1] = eulerStepNumpy_Pendulum(pred_inv[i], tau*gamma[0], extraParams)

# Secon and third time
pred_inv2 = np.zeros([M, d])
pred_inv3 = np.zeros([M, d])
for i in range(M):
    pred_inv2[i] = eulerStepNumpy_Pendulum(pred_inv[i+1], tau*gamma[1], extraParams)
    pred_inv3[i] = eulerStepNumpy_Pendulum(pred_inv2[i], tau*gamma[2], extraParams)
    
pred_inv3 = torch.from_numpy(np.float64(pred_inv3)).reshape(M, 1, d)
with torch.no_grad():
    pred, _ = model(pred_inv3, Tau)

pred = pred.reshape((M, d)).numpy()

pred_triple = np.zeros([M+1, d])
pred_triple[0] = x0
pred_triple[1::] = pred