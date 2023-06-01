import numpy as np
import matplotlib.pyplot as plt
from analysis_generalfunc import generate_starting_points, analytical_pred_Tend
from TrainingData.general_problems import Kepler, Pendulum
from TrainingData import verlet8

# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

area_Pendulum = np.array([[[-1.5, 2]], [[-1.5, 1.5]]]) # Same as area from training data
seed = 0
nr_trajects = 1
x0 = generate_starting_points(area_Pendulum, seed, nr_trajects)
tau = 0.1
Tend =10
extraParams = None

exacts = []
d = len(x0[0])
D = int(d/2)
M = int(Tend/tau)
for i in range(nr_trajects):
    exact = np.zeros([M+1, d])
    exact[0] = x0[i]

    for i in range(M):
        exact[i+1] = verlet8.eight_orderPrecise(exact[i], tau, D, Pendulum, extraParams)


    exacts.append(exact)

fig, ax1 = plt.subplots(figsize=(9, 6.5))
for i in range(nr_trajects):
    ax1.plot(exacts[i][:, 0], exacts[i][:, 1])