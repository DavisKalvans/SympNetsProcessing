import general_trainingVerlSym
import numpy as np
import torch

### HarmOsc Verlet settings
problem = 'Kepler'
method = 'Verlet'
device = 'cpu'
N = 40 # Training data
M = 100 # Testing data
tau = 0.1 # Time step
nL = 2 # Layers
nN = 4 # Width of each layer
nM = 0 # Seed for model parameter initialization
epochs = 1_000 # Only use thousands here
extraParams = None # Extra parameters for problems that need it, eg, HarmOsc has omega; None by default

### Areas for mult predictions/convergence; starting points for one trajectory prediction/convergence
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
# Area works like this area[0] is all min and max values of each q coordinate
# and area[1] is all min and max values of each p coordinate.
# E.g. for Kepler q1 [-1.5, 1.5]; q2 [-1.5, 1.5]; p1 [-1, 1]; p2 [-1, 1].

linear = False # Use a linear activation function
learning_rate = 1e-2
sigma = 1 # Parameter initialization weights
sch = False # Use scheduling for learning rate
eta1 = 1e-1 # For scheduling (starting learning rate)
eta2 = 1e-3 # For scheduling (ending learning rate)

if problem == "Kepler":
    x0 = x0_Kepler
    area = area_Kepler
    dim = 4
elif problem == "Pendulum":
    x0 = x0_Pendulum
    area = area_Pendulum
    dim = 2

data = (N, M, tau) # Training data N, testing data M, time step tau
trainParams = (learning_rate, N, epochs, sch, eta1, eta2, linear, sigma) # Learning rate, batch size, epochs, scheduling, eta1, eta2, linear
predParams = (10000, x0, False) # Time steps, x0, best
plotParams = (1000, tau, x0, False) # Time step to plot until, dimension, best
convParams = (x0, 10, 0, 5, dim, False) # x0, Tend, lower power, upper power, dim, best
predParamsMult = (1000, tau, area, 0, 10, False) # Time steps, area for starting points, timestep to use, seed, 
#nr of trajectories, use the best model or not
convParamsMult = (10, 0, 5, dim, area, 0, 10, False) # Tend, lower power, upper power, area for starting points,
#seed, nr of trajectories, use the best model or not

# Loop trough combinations of layer width
for nL in [2]: # Layers
    for nN in [4]: # Width
        for nM in [0, 1]: # Seeds to use
            # Train model
            general_trainingVerlSym.train_model(problem, method, device, data, trainParams, nL, nN, nM, extraParams)
            # Calculate and plot prediction errors for one specific trajectory x0
            general_trainingVerlSym.model_errors(problem, method, device, data, trainParams, nL, nN, nM, plotParams, extraParams)
            # Calculate and plot prediction error convergence for one specific trajectory x0
            general_trainingVerlSym.model_conv(problem, method, device, data, trainParams, nL, nN, nM, convParams, extraParams)
            # Calculate and plot prediction error average for many trajectories
            general_trainingVerlSym.model_multTrajectErrors(problem, method, device, data, trainParams, nL, nN, nM, predParamsMult, extraParams)
            # Calculate and plot prediction error convergence average for many trajectories
            general_trainingVerlSym.model_multTrajectConv(problem, method, device, data, trainParams, nL, nN, nM, convParamsMult, extraParams)

            # Do everything with best model too
            #predParams_best = (10000, x0, True) # Time steps, x0, best
            #plotParams_best = (1000, dim, True) # Time step to plot until, dimension, best
            #convParams_best = (x0, 10, 1, 14, dim, True) # x0, Tend, lower power, upper power, dim, best
            #predParamsMult_best = (1000, tau, area, 0, 10, True)
            #convParamsMult_best = (10, 0, 5, dim, area, 0, 10, True) # Tend, lower power, upper power, area for starting points,
            #general_trainingVerlSym.model_predictions(problem, method, device, data, trainParams, nL, nN, nM, predParams_best, extraParams)
            #general_trainingVerlSym.model_errors(problem, method, data, trainParams, nL, nN, nM, plotParams_best, extraParams)
            #general_trainingVerlSym.model_multTrajectErrors(problem, method, device, data, trainParams, nL, nN, nM, predParamsMult_best, extraParams)
            #general_trainingVerlSym.model_multTrajectConv(problem, method, device, data, trainParams, nL, nN, nM, convParamsMult_best, extraParams)