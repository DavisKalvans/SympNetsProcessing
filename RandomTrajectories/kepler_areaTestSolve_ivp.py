import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp   
from TrainingData.kepler import Kepler, Kepler_H, Kepler_L
from analysis_generalfunc import generate_starting_points, analytical_pred_Tend


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
tau = 0.1
Tend = 50
extraParams = None
M = int(Tend/tau)
tm = np.linspace(0, Tend, M+1)

exacts = []
for i in range(nr_trajects):
    exact = solve_ivp(Kepler, [0, Tend], x0[i], t_eval=tm, method='RK45', rtol = 1e-12, atol = 1e-12)
    exacts.append(exact.y)

fig, ax1 = plt.subplots(figsize=(9, 6.5))
for i in range(nr_trajects):
    ax1.plot(exacts[i][0, :], exacts[i][1, :])


fig, ax2 = plt.subplots(figsize=(9, 6.5))
for i in range(nr_trajects):
    ax2.plot(exacts[i][2, :], exacts[i][3, :])