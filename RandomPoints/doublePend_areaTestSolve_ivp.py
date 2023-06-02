import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp   
from TrainingData.DoublePend import DoublePendulum, DoublePendulum_H
from analysis_generalfunc import generate_starting_points, analytical_pred_Tend


# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

# For training dataset probably np.array([[[-1.5, 1.5], [-2, 2], [-2, 2], [-1.5, 1.5]]])
area_DoublePend = np.array([[[-1, 1], [-1, 1]], [[-1, 1], [-1, 1]]]) # Same as area from training dataseed = 0
seed = 0
nr_trajects = 1
x0 = generate_starting_points(area_DoublePend, seed, nr_trajects)
tau = 0.1
Tend = 100
extraParams = None
M = int(Tend/tau)
tm = np.linspace(0, Tend, M+1)

exacts = []
for i in range(nr_trajects):
    exact = solve_ivp(DoublePendulum, [0, Tend], x0[i], t_eval=tm, method='RK45', rtol = 1e-12, atol = 1e-12)
    exacts.append(exact.y)

fig, ax1 = plt.subplots(figsize=(9, 6.5))
for i in range(nr_trajects):
    ax1.plot(exacts[i][0, :], exacts[i][2, :])
    ax1.plot(exacts[i][0, 0], exacts[i][2, 0], marker ='o')

fig, ax2 = plt.subplots(figsize=(9, 6.5))
for i in range(nr_trajects):
    ax2.plot(exacts[i][1, :], exacts[i][3, :])
    ax2.plot(exacts[i][1, 0], exacts[i][3, 0], marker = 'o')