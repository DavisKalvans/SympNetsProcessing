# Okay, so when we take all the training data as one batch, then our model doesn't learn much and gets stuck
# But for some reason, when the batch size is set to 1, then it starts learning.
# That's why this test script is meant to investigate and find out what's wrong with the current training loop implementation.

### Conclusions - no torch.matmul when you need a dot product actually ...

import numpy as np
import time
import torch
import copy
from torch import nn
from NeuralNetwork.custom_dataset import CustomDataset
from NeuralNetwork.mySequential import mySequential
from NeuralNetwork.symp_module_class import SympGradModule
from NeuralNetwork.training_class import train_loop, test_loop
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

# Find device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
torch.set_num_threads(1)

# Load training data
N = 10
M = 5
tau = 0.1
tau_txt = str(tau).replace('.', '')

data = np.load(f'TrainingData/SavedTrainingData/HarmOsc/HarmOscRandN{N}M{M}ConstTau{tau_txt}.npz')
x_train = torch.from_numpy(np.float32(data['train_X'])).to(device)  
y_train = torch.from_numpy(np.float32(data['train_Y'])).to(device) 
tau_train = torch.from_numpy(np.float32(data['train_Tau'])).to(device)
x_test = torch.from_numpy(np.float32(data['test_X'])).to(device)
y_test = torch.from_numpy(np.float32(data['test_Y'])).to(device)
tau_test = torch.from_numpy(np.float32(data['test_Tau'])).to(device)
omega = torch.tensor(0.5, dtype=torch.float32, device=device)

# Dimension of the problem
D = x_train.shape[2]
d = int(D/2)

nL = 2 # Layers 
nN = 4  # Width of each layer


layers = []
for n in range(nL):
    layers.append(SympGradModule(d, nN, nL))
    model = mySequential(*layers).to(device)

print(model)

### This is how the train loop worked
# inverse, _ = model.back(X, tau) # Pass trough inverse model
        
# # Need to do the numerical method now (symplectic Euler)
# XX = torch.zeros((X.size(dim=0), X.size(dim=1), X.size(dim=2)), dtype=torch.float32)
            
        
# a = inverse[:, 0, 1] - torch.matmul(omega**2*tau.T, inverse[:, 0, 0])
# b = inverse[:, 0, 0] + a[0,  :]*tau.T
# a = a.reshape((1, 1, X.size(dim=0)))
# XX[:, 0, 1] = a
# XX[:, 0, 0] = b

### Let's do one point at a time
predictions = []
inverse1 = []

with torch.no_grad():
    for i in range(N):
        X = x_train[i].reshape((1, 1, 2))
        tau = tau_train[i].reshape((1, 1, 1))
        inverse, _ = model.back(X, tau)
        inverse1.append(inverse)

        XX = torch.zeros((X.size(dim=0), X.size(dim=1), X.size(dim=2)), dtype=torch.float32)

        a = inverse[:, 0, 1] - torch.matmul(omega**2*tau.T, inverse[:, 0, 0])
        b = inverse[:, 0, 0] + a[0,  :]*tau.T
        a = a.reshape((1, 1, X.size(dim=0)))
        XX[:, 0, 1] = a
        XX[:, 0, 0] = b

        pred, _ = model(XX, tau) # Pass trough original model

        predictions.append(pred)


### And let's do the whole batch for comparison

with torch.no_grad():
    inverse, _ = model.back(x_train, tau_train)

    XX = torch.zeros((x_train.size(dim=0), x_train.size(dim=1), x_train.size(dim=2)), dtype=torch.float32)

    #a = inverse[:, 0, 1] - torch.matmul(omega**2*tau_train.T, inverse[:, 0, 0])
    a = inverse[:, 0, 1] - omega**2*tau_train.T*inverse[:, 0, 0]
    b = inverse[:, 0, 0] + a[0,  :]*tau_train.T
    a = a.reshape((1, 1, x_train.size(dim=0)))
    XX[:, 0, 1] = a
    XX[:, 0, 0] = b

    pred_batch, _ = model(XX, tau_train) # Pass trough original model



