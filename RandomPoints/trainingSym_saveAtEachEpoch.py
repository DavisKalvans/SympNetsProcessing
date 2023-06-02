import general_trainingVerlSym
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
### HarmOsc Verlet settings
problem = 'Pendulum'
method = 'Verlet'
device = 'cpu'
N = 40 # Training data
M = 20 # Testing data
tau = 0.1 # Time step
nL = 2 # Layers
nN = 4 # Width of each layer
nM = 0 # Seed for model parameter initialization
epochs = 50_000
extraParams = None # Extra parameters for problems that need it, eg, HarmOsc has omega; None by default
x0_HarmOsc = (0.3, 0.5)
x0_Pendulum = (0.8, 0.5)
e = 0.6
q1 = 1 - e
q2 = 0
p1 = 0
p2 = np.sqrt((1+e)/(1-e))
x0_Kepler = (q1, q2, p1, p2)
dim = 2
linear = True
learning_rate = 1e-3
sch = True
eta1 = 1
eta2 = 1e-5

data = (N, M, tau) # Training data N, testing data M, time step tau
trainParams = (learning_rate, N, epochs, sch, eta1, eta2, linear) # Learning rate, batch size, epochs, scheduling, eta1, eta2, linear
predParams = (1000, x0_Pendulum, False) # Time steps, x0, best
plotParams = (1000, dim, False) # Time step to plot until, dimension, best
convParams = (x0_Pendulum, 10, 3, 14, dim, False) # x0, Tend, lower power, upper power, dim, best

# Loop trough combinations of layer width
for nL in [2]:
    for nN in [4]:
        #general_trainingVerlSym.train_modelSaveAtEachEpoch(problem, method, device, data, trainParams, nL, nN, nM, extraParams)
        general_trainingVerlSym.model_errorsAtEachEpoch(problem, method, data, trainParams, nL, nN, nM, predParams, 100, extraParams)