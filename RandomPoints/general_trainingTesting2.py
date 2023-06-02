import numpy as np
import torch
# Do the numpy and torch functions for one step of numerical verlet match up?
# All of them seem to be fine, errors in the range of 6-8 order

### Testing pendulum
from general_training import verletStep_Pendulum, verletStepNumpy_Pendulum

tau = 0.1
X0 = ((0.3, 0.5), (0.4, 0.2), (0.7, 0.1), (0.1, 0.2))

numpy_results = np.zeros((len(X0), 2))

for i in range(len(X0)):
    numpy_results[i] = verletStepNumpy_Pendulum(X0[i], tau)

numpy_results = numpy_results.reshape((len(X0), 1, 2))

X0_torch = torch.tensor(X0, dtype=torch.float64).reshape((len(X0), 1, 2))
Tau = torch.tensor([[[tau]]])
torch_results = verletStep_Pendulum(X0_torch, Tau)
torch_results = torch_results.numpy()

diff = torch_results-numpy_results
print(f'Max error in Penudlum was {diff.max()}.')


### Testing HarmOsc
from general_training import verletStep_HarmOsc, verletStepNumpy_HarmOsc

tau = 0.1
X0 = ((0.3, 0.5), (0.4, 0.2), (0.7, 0.1), (0.1, 0.2))
omega = 0.5

numpy_results = np.zeros((len(X0), 2))

for i in range(len(X0)):
    numpy_results[i] = verletStepNumpy_HarmOsc(X0[i], tau, omega)

numpy_results = numpy_results.reshape((len(X0), 1, 2))

X0_torch = torch.tensor(X0, dtype=torch.float64).reshape((len(X0), 1, 2))
Tau = torch.tensor([[[tau]]])
torch_results = verletStep_HarmOsc(X0_torch, Tau, omega)
torch_results = torch_results.numpy()


diff = torch_results-numpy_results
print(f'Max error in HarmOsc was {diff.max()}.')


### Testing Kepler
from general_training import verletStep_Kepler, verletStepNumpy_Kepler

tau = 0.1
X0 = ((0.1, 0, 0.3, 0.7), (0, 0.1, 0.4, 0.8), (0.1, 0.1, 0.3, 0.2), (0.2, 0, 0.8, 0.4))

numpy_results = np.zeros((len(X0), 4))

for i in range(len(X0)):
    numpy_results[i] = verletStepNumpy_Kepler(X0[i], tau)

numpy_results = numpy_results.reshape((len(X0), 1, 4))

X0_torch = torch.tensor(X0, dtype=torch.float64).reshape((len(X0), 1, 4))
Tau = torch.tensor([[[tau]]])
torch_results = verletStep_Kepler(X0_torch, Tau)
torch_results = torch_results.numpy()


diff = torch_results-numpy_results
print(f'Max error in Kepler was {diff.max()}.')