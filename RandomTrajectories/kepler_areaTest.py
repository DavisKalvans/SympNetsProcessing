import numpy as np
import matplotlib.pyplot as plt
from analysis_generalfunc import generate_starting_points, analytical_pred_Tend
from TrainingData.general_problems import Kepler, Pendulum
from TrainingData import verlet8


# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

def generate_starting_pointsKepler(seed, nr_trajects):
    np.random.seed(seed)
    x0 = []

    for i in range(nr_trajects):
        x0.append([])
        e = np.random.rand()*0.8 +0.1 # in range [0.1, 0.9]
        x0[-1].append(1-e)
        x0[-1].append(0)
        x0[-1].append(0)
        p2 = np.sqrt((1+e)/(1-e))
        x0[-1].append(p2)

    return x0

e = 0.6
q1 = 1 - e
q2 = 0
p1 = 0
p2 = np.sqrt((1+e)/(1-e))
x0_Kepler = (q1, q2, p1, p2)


area_Kepler = np.array([[[-0.5, 0.5], [-0.5, 0.5]], [[-0.5, 0.5], [-0.5, 0.5]]]) # Same as area from training dataseed = 0
seed = 901
nr_trajects = 10
x0 = generate_starting_pointsKepler(seed, nr_trajects)
#x0[0] = x0_Kepler
tau = 0.01
Tend = 50
extraParams = None



exacts = []
d = len(x0[0])
D = int(d/2)
M = int(Tend/tau)
for i in range(nr_trajects):
    exact = np.zeros([M+1, d])
    exact[0] = x0[i]

    for i in range(M):
        exact[i+1] = verlet8.eight_orderPrecise(exact[i], tau, D, Kepler, extraParams)


    exacts.append(exact)

fig, ax1 = plt.subplots(figsize=(9, 6.5))
for i in range(nr_trajects):
    ax1.plot(exacts[i][:, 0], exacts[i][:, 1])


fig, ax2 = plt.subplots(figsize=(9, 6.5))
for i in range(nr_trajects):
    ax2.plot(exacts[i][:, 2], exacts[i][:, 3])