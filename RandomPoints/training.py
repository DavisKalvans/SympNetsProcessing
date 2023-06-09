import general_training
import numpy as np

### HarmOsc Verlet settings
problem = 'Kepler'
method = 'Euler'
device = 'cpu'
N = 40 # Training data
M = 100 # Testing data
tau = 0.01 # Time step
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

if problem == "Kepler":
    dim = 4
    x0 = x0_Kepler
    area = area_Kepler
elif problem == "Pendulum":
    dim = 2
    x0 = x0_Pendulum
    area = area_Pendulum

linear = False # Use a linear activation function
learning_rate = 1e-2
sch = False # Use scheduling for learning rate
eta1 = 1e-1 # For scheduling (starting learning rate)
eta2 = 1e-2 # For scheduling (ending learning rate)

data = (N, M, tau) # Training data N, testing data M, time step tau

trainParams = (learning_rate, N, epochs, sch, eta1, eta2, linear) # Learning rate, batch size, epochs, scheduling, eta1, eta2, linear
predParams = (1000, tau, x0, False) # Time steps, x0, best
convParams = (x0, 10, 0, 5, dim, False) # x0, Tend, lower power, upper power, dim, best
predParamsMult = (1000, tau, area, 0, 10, False) # Time steps, area for starting points, timestep to use, seed, 
#nr of trajectories, use the best model or not
convParamsMult = (10, 0, 5, dim, area, 0, 10, False) # Tend, lower power, upper power, area for starting points,
#seed, nr of trajectories, use the best model or not

# Loop trough combinations of layer count and their width
for nL in [2]: # Layers
    for nN in [4]: # Width
        for nM in [0, 1]: # Seeds to use
            # Train model
            general_training.train_model(problem, method, device, data, trainParams, nL, nN, nM, extraParams)
            # Calculate and plot prediction errors for one specific trajectory x0
            general_training.model_errors(problem, method, device, data, trainParams, nL, nN, nM, predParams, extraParams)
            # Calculate and plot prediction error convergence for one specific trajectory x0
            general_training.model_conv(problem, method, device, data, trainParams, nL, nN, nM, convParams, extraParams)
            # Calculate and plot prediction error average for many trajectories
            general_training.model_multTrajectErrors(problem, method, device, data, trainParams, nL, nN, nM, predParamsMult, extraParams)
            # Calculate and plot prediction error convergence average for many trajectories
            general_training.model_multTrajectConv(problem, method, device, data, trainParams, nL, nN, nM, convParamsMult, extraParams)

            # Do everything with best model too
            #predParams_best = (1000, x0_Pendulum, True) # Time steps, x0, best
            #convParams_best = (x0_Pendulum, 10, 0, 5, dim, True) # x0, Tend, lower power, upper power, dim, best
            #predParamsMult_best = (1000, tau, area_Pendulum, 0, 10, True)
            #convParamsMult_best = (10, 0, 5, dim, area_Pendulum, 0, 10, True)
            #general_training.model_errors(problem, method, device, data, trainParams, nL, nN, nM, predParams_best, extraParams)
            #general_training.model_conv(problem, method, device, data, trainParams, nL, nN, nM, convParams_best, extraParams)
            #general_training.model_multTrajectErrors(problem, method, device, data, trainParams, nL, nN, nM, predParamsMult_best, extraParams)
            #general_training.model_multTrajectConv(problem, method, device, data, trainParams, nL, nN, nM, convParamsMult_best, extraParams)