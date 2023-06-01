import numpy as np
import time
import torch
import copy
from torch import nn
from NeuralNetwork.custom_dataset import CustomDataset
from NeuralNetwork.mySequential import mySequential
from NeuralNetwork.symp_module_class import SympGradModule

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

data = np.load(f'TrainingData/SavedTrainingData/Kepler/KeplerRandN{N}M{M}ConstTau{tau_txt}.npz')
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

# Custom Dataset 
training_data = CustomDataset(x_train, y_train, tau_train)
testing_data = CustomDataset(x_test, y_test, tau_test)

# Training parameter values
learning_rate = 1e-4
batch_size = N
epochs = 50_000
epochs_th = str(epochs/1000).replace('.0', '')
sch = True
eta1 = 1e-3
eta2 = 1e-5

gamma = np.exp(np.log(eta2/eta1)/epochs)
if sch:
    learning_rate = eta1
    eta1_txt = str(np.log10(eta1)).replace('-', '').replace('.0', '')
    eta2_txt = str(np.log10(eta2)).replace('-', '').replace('.0', '')
else:
    eta1_txt = str(np.log10(learning_rate)).replace('-', '').replace('.0', '')




nL = 2 # Layers
nN = 2 # Width of each layer
nM = 0 # Seed

layers = []
for n in  range(nL):
    layers.append(SympGradModule(d, nN, nL))

model = mySequential(*layers).to(device)
print(model)
                        
                # Set random seed to generate the same initial parameter values
torch.manual_seed(nM)
sigma = np.sqrt(0.01) # Does this being sqrt(0.01) ruin everything?
for param in model.parameters():
    param.data = sigma*torch.randn(param.shape)

model = model.to(device)

inverse, _ = model.back(x_train, tau_train)
tau_vec = tau_train.reshape(tau_train.size(0))

XX = torch.zeros_like(x_train, dtype = torch.float32)
q1 = inverse[:, 0, 0] +inverse[:, 0, 2]*tau_vec
q2 = inverse[:, 0, 1] +inverse[:, 0, 3]*tau_vec

d = torch.pow(torch.pow(q1, 2)+torch.pow(q2, 2), 1.5)

p1 = inverse[:, 0, 2] -q1/d*tau_vec
p2 = inverse[:, 0, 3] -q2/d*tau_vec

XX[:, 0, 0] = q1
XX[:, 0, 1] = q2
XX[:, 0, 2] = p1
XX[:, 0, 3] = p2

pred, _ = model(XX, tau_train)
