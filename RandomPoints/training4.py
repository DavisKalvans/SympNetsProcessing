import general_training
import numpy as np

### HarmOsc Verlet settings
problem = 'Kepler'
method = 'Verlet'
device = 'cpu'
N = 320 # Training data
M = 100 # Testing data
tau = 0.01 # Time step
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
learning_rate = 1e-4
sch = True
eta1 = 1e-1
eta2 = 1e-3

data = (N, M, tau) # Training data N, testing data M, time step tau

trainParams = (learning_rate, N, epochs, sch, eta1, eta2, linear) # Learning rate, batch size, epochs, scheduling, eta1, eta2, linear
predParams = (10000, x0_Pendulum, False) # Time steps, x0, best
plotParams = (1000, dim, False) # Time step to plot until, dimension, best
convParams = (x0_Pendulum, 10, 1, 14, dim, False) # x0, Tend, lower power, upper power, dim, best

# Loop trough combinations of layer count and their width
for nL in [2, 4, 8]:
    for nN in [16]:
        general_training.train_model(problem, method, device, data, trainParams, nL, nN, nM, extraParams)
        general_training.model_predictions(problem, method, device, data, trainParams, nL, nN, nM, predParams, extraParams)
        general_training.model_errors(problem, method, data, trainParams, nL, nN, nM, plotParams, extraParams)
        general_training.model_conv(problem, method, device, data, trainParams, nL, nN, nM, convParams, extraParams)

        # Do everything with best model too
        predParams_best = (10000, x0_Pendulum, True) # Time steps, x0, best
        plotParams_best = (1000, dim, True) # Time step to plot until, dimension, best
        convParams_best = (x0_Pendulum, 10, 1, 14, dim, True) # x0, Tend, lower power, upper power, dim, best
        general_training.model_predictions(problem, method, device, data, trainParams, nL, nN, nM, predParams_best, extraParams)
        general_training.model_errors(problem, method, data, trainParams, nL, nN, nM, plotParams_best, extraParams)
        general_training.model_conv(problem, method, device, data, trainParams, nL, nN, nM, convParams_best, extraParams)