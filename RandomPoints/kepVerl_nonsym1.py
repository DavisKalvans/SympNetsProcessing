import general_training
import numpy as np

### HarmOsc Verlet settings
problem = 'Kepler'
method = 'Verlet'
device = 'cpu'
N = 40 # Training data
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
area_Pendulum = np.array([[[-1.5, 2]], [[-1.5, 1.5]]]) # Same as area from training data
area_HarmOsc = np.array([[[-1.4, 1.4]], [[-0.8, 0.8]]]) # Same as area from training data
area_Kepler = np.array([[[-1.5, 1.5], [-1.5, 1.5]], [[-1, 1], [-1, 1]]]) # Same as area from training data


if problem == "Kepler":
    dim = 4
    x0 = x0_Kepler
    area = area_Kepler
elif problem == "Pendulum":
    dim = 2
    x0 = x0_Pendulum
    area = area_Pendulum

linear = False
learning_rate = 1e-2
sch = True
eta1 = 1e-1
eta2 = 1e-3

data = (N, M, tau) # Training data N, testing data M, time step tau

trainParams = (learning_rate, N, epochs, sch, eta1, eta2, linear) # Learning rate, batch size, epochs, scheduling, eta1, eta2, linear
predParams = (10000, x0, False) # Time steps, x0, best
plotParams = (1000, dim, False) # Time step to plot until, dimension, best
convParams = (x0, 10, 0, 5, dim, False) # x0, Tend, lower power, upper power, dim, best
predParamsMult = (1000, tau, area, 0, 10, False) # Time steps, area for starting points, timestep to use, seed, 
#nr of trajectories, use the best model or not
convParamsMult = (10, 0, 5, dim, area, 0, 10, False) # Tend, lower power, upper power, area for starting points,
#seed, nr of trajectories, use the best model or not

# Loop trough combinations of layer count and their width
for nL in [2]:
    for nN in [4]:
        general_training.train_model(problem, method, device, data, trainParams, nL, nN, nM, extraParams)
        #general_training.model_predictions(problem, method, device, data, trainParams, nL, nN, nM, predParams, extraParams)
        #general_training.model_errors(problem, method, data, trainParams, nL, nN, nM, plotParams, extraParams)
        #general_training.model_conv(problem, method, device, data, trainParams, nL, nN, nM, convParams, extraParams)
        general_training.model_multTrajectErrors(problem, method, device, data, trainParams, nL, nN, nM, predParamsMult, extraParams)
        general_training.model_multTrajectConv(problem, method, device, data, trainParams, nL, nN, nM, convParamsMult, extraParams)

        # Do everything with best model too
        #predParams_best = (10000, x0, True) # Time steps, x0, best
        #plotParams_best = (1000, dim, True) # Time step to plot until, dimension, best
        #convParams_best = (x0, 10, 0, 5, dim, True) # x0, Tend, lower power, upper power, dim, best
        #general_training.model_predictions(problem, method, device, data, trainParams, nL, nN, nM, predParams_best, extraParams)
        #general_training.model_errors(problem, method, data, trainParams, nL, nN, nM, plotParams_best, extraParams)
        #general_training.model_conv(problem, method, device, data, trainParams, nL, nN, nM, convParams_best, extraParams)