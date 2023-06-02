import numpy as np
import torch

#-------------------------------------------------------------
### Test 1
# See the precision differences between torch and numpy floats
# They match up, but float 32 seems to be accurate up to approx 7 digits;
# float 64 is accurate up to approx 16 digits

# frac_numpy = np.array(1/3)
# frac_numpy32 = np.array(1/3, dtype = np.float32)
# frac_torch = torch.tensor(1/3).float()
# frac_torch64 = torch.tensor(1/3, dtype = torch.float64)

# print('%.60f - Numpy and float64' % (frac_numpy))
# print('%.60f - Torch and torch.float64' % (frac_torch64))
# print('%.60f - Numpy and float32' % (frac_numpy32))
# print('%.60f - Torch and torch.float32' % (frac_torch))
#-------------------------------------------------------------



#-------------------------------------------------------------
### Test 2
# See if we can wrangle pytorch to do matmul with float64
# It can most certainly use float64 and achieve the same exact accuracy as you'd expect

# A = np.matrix([[1/3, 1/3], [1/3, 1/3]])
# A2 = np.matmul(A, A)

# B = torch.tensor([[1/3, 1/3], [1/3, 1/3]])
# B2 = torch.matmul(B, B)

# C = torch.tensor([[1/3, 1/3], [1/3, 1/3]], dtype = torch.float64)
# C2 = torch.matmul(C, C)


# print('%.60f - Numpy and float64' % (A2[0, 0]))
# print('%.60f - Torch and torch.float64' % (C2[0, 0]))
# print('%.60f - Torch and torch.float32' % (B2[0, 0]))
#-------------------------------------------------------------


#-------------------------------------------------------------
### Test 3
# Create the simplest symplectic neural network and try to do a forward pass with float64
# Can do the forward pass with no problems with float64 accuracy

from NeuralNetwork.symp_module_class import SympGradModule
from NeuralNetwork.mySequential import mySequential

nL = 2; nN = 4
torch.manual_seed(0)
layers = []
for n in range(nL):
    layers.append(SympGradModule(1, nN, nL))

model = mySequential(*layers)

#x32 = torch.tensor((0.8, 0.5), dtype = torch.float32).reshape((1, 1, 2))
#tau32 = torch.tensor([[[0.1]]], dtype = torch.float32)

#rez32, _ = model(x32, tau32)

x64 = torch.tensor((0.8, 0.5), dtype = torch.float64).reshape((1, 1, 2))
tau64 = torch.tensor([[[0.1]]], dtype = torch.float64)

rez64, _ = model(x64, tau64)

print('%.60f - Torch and torch.float64' % (rez64[0, 0, 1]))
#print('%.60f - Torch and torch.float32' % (rez32[0, 0, 1]))
#-------------------------------------------------------------


#-------------------------------------------------------------
### Test 4
# Loading in saved training data
# Everything is fine with this implementation

# npz_file = np.load(f'TrainingData/SavedTrainingData/Pendulum/PendulumRandN10M5ConstTau01.npz')

# x_train = np.float64(npz_file['train_X'])
# x_train_torch = torch.from_numpy(np.float64(npz_file['train_X']))

# print('%.60f - Numpy and float64' % (x_train[0, 0, 0]))
# print('%.60f - Torch and float64' % (x_train_torch[0, 0, 0]))

#-------------------------------------------------------------


# a = torch.tensor(1/3, dtype = torch.float64)
# a2 = torch.pow(a, 2)