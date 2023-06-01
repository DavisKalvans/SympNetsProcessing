import numpy as np
import torch
from torch import nn
from symp_module_class import SympGradModule
from mySequential import mySequential

# find device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
torch.set_num_threads(1)

nL = 16 # Number of layers
nN = 64 # Width of each layer
d = 1 # Half the dimension D of the problem (HarmOsc has D=2)

# Create the neural network model
layers = []
for n in range(nL):
    layers.append(SympGradModule(d, nN, nL))
    model = mySequential(*layers).to(device)

print(model)

# Set random seed to generate the same initial parameter values
nM = 0
torch.manual_seed(nM)
sigma = np.sqrt(0.01)
for param in model.parameters():
    param.data = sigma*torch.randn(param.shape)

with torch.no_grad():
    a = torch.tensor([0.5, 3.1], dtype=torch.float32, device=device).reshape((1, 1, 2))
    tau = torch.tensor([50], dtype=torch.float32, device=device)
    b, _ = model(a, tau)
    c, _ = model.back(b, tau)


difference = c-a
print(a)
print(b)
print(c)
print(f"Difference is {difference}")