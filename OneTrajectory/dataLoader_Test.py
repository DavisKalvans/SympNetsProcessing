import numpy as np
import torch
import copy
from torch import nn
from torch.utils.data import DataLoader
from NeuralNetwork.custom_dataset import CustomDataset

N = 10
M = 5
tau = 0.01
problem = 'Kepler'
device = 'cpu'


tau_txt = str(tau).replace('.', '')
npz_file = np.load(f'TrainingData/SavedTrainingData/{problem}/{problem}RandN{N}M{M}ConstTau{tau_txt}.npz')

x_train = torch.from_numpy(np.float64(npz_file['train_X'])).to(device)  
y_train = torch.from_numpy(np.float64(npz_file['train_Y'])).to(device) 
tau_train = torch.from_numpy(np.float64(npz_file['train_Tau'])).to(device)
x_test = torch.from_numpy(np.float64(npz_file['test_X'])).to(device)
y_test = torch.from_numpy(np.float64(npz_file['test_Y'])).to(device)
tau_test = torch.from_numpy(np.float64(npz_file['test_Tau'])).to(device)

training_data = CustomDataset(x_train, y_train, tau_train)
testing_data = CustomDataset(x_test, y_test, tau_test)

train_dataloader = DataLoader(training_data, batch_size=N)
test_dataloader = DataLoader(testing_data, batch_size=M)

# Batch and enumerate ar completely useless
for (X, y, Tau) in test_dataloader:
    x0 = X  
    x1 = copy.deepcopy(X)

x1[0, 0, 0] += 3
x1[0, 0, 1] += 4
loss_fn = nn.MSELoss(reduction = 'sum')
a = loss_fn(x0, x1)/M
print(a)