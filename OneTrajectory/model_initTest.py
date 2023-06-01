import numpy as np
import torch
import copy
from torch import nn
from torch.utils.data import DataLoader
from NeuralNetwork.custom_dataset import CustomDataset
from NeuralNetwork.symp_module_class import SympGradModule, LinSympGradModule
from NeuralNetwork.mySequential import mySequential
from general_trainingVerlSym import verletStep_Kepler

N = 320
M = 100
tau = 0.01
problem = 'Kepler'
device = 'cpu'

nL = 2
nN = 4
nM = 0

d = 2 # Half dimension?
sigma = 1 # Parameter initialization weight
torch.manual_seed(nM)
layers = []
for n in  range(nL):
    layers.append(SympGradModule(d, nN, nL, sigma))

model = mySequential(*layers).to(device)

#for param in model.parameters():
#        print(param)


###
tau_txt = str(tau).replace('.', '')
npz_file = np.load(f'TrainingData/SavedTrainingData/{problem}/{problem}RandN{N}M{M}ConstTau{tau_txt}.npz')

x_train = torch.from_numpy(np.float64(npz_file['train_X'])).to(device)  
y_train = torch.from_numpy(np.float64(npz_file['train_Y'])).to(device) 
tau_train = torch.from_numpy(np.float64(npz_file['train_Tau'])).to(device)
x_test = torch.from_numpy(np.float64(npz_file['test_X'])).to(device)
y_test = torch.from_numpy(np.float64(npz_file['test_Y'])).to(device)
tau_test = torch.from_numpy(np.float64(npz_file['test_Tau'])).to(device)

# Just kernel
y = verletStep_Kepler(x_train, tau_train, None)

# Processed
inverse, _ = model.back(x_train, torch.pow(tau_train, 2))
pred_inv = verletStep_Kepler(inverse, tau_train, None)
y_pred, _ = model(pred_inv, torch.pow(tau_train, 2))

loss_fn = nn.MSELoss()
loss_kernel = loss_fn(y_train, y)
print(loss_kernel)

loss = loss_fn(y_train, y_pred)
print(loss)

