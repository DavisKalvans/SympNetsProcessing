import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp   
from TrainingData.DoublePend import DoublePendulum, DoublePendulum_H
from analysis_generalfunc import generate_starting_points, analytical_pred_Tend, numeric_predict_Tend
import torch

def verlet_stepNumpy_DoublePend(x, tau, params = None):
    tau_half = tau/2

    q1_add = (x[2] -x[3]*np.cos(x[0]-x[1]))/(1 +np.sin(x[0]-x[1])**2)
    q2_add = (-x[2]*np.cos(x[0]-x[1]) +2*x[3])/(1 +np.sin(x[0]-x[1])**2)
    q1_half = x[0] +q1_add*tau_half
    q2_half = x[1] +q2_add*tau_half

    h1 = x[2]*x[3]*np.sin(q1_half-q2_half)
    h1 /= 1 +np.sin(q1_half-q2_half)**2
    h2 = x[2]**2 +2*x[3]**2 -2*x[2]*x[3]*np.cos(q1_half-q2_half)
    h2 /= 2*(1 +np.sin(q1_half-q2_half)**2)**2
    p1_add = -2*np.sin(q1_half) -h1 +h2*np.sin(2*q1_half-2*q2_half)
    p2_add = -np.sin(q2_half) +h1 -h2*np.sin(2*q1_half-2*q2_half)
    p1 = x[2] +p1_add*tau
    p2 = x[3] +p2_add*tau

    q1_add = p1-p2*np.cos(q1_half-q2_half)
    q1_add /= 1 +np.sin(q1_half-q2_half)**2
    q2_add = -p1*np.cos(q1_half-q2_half) +2*p2
    q2_add /= 1 +np.sin(q1_half-q2_half)**2
    q1 = q1_half +q1_add*tau_half
    q2 = q2_half +q2_add*tau_half

    return (q1, q2, p1, p2)

def euler_stepNumpy_DoublePend(x, tau, params = None):
    q1_add = (x[2] -x[3]*np.cos(x[0]-x[1]))/(1 +np.sin(x[0]-x[1])**2)
    q2_add = (-x[2]*np.cos(x[0]-x[1]) +2*x[3])/(1 +np.sin(x[0]-x[1])**2)
    q1 = x[0] +q1_add*tau
    q2 = x[1] +q2_add*tau

    h1 = x[2]*x[3]*np.sin(q1-q2)
    h1 /= 1 +np.sin(q1-q2)**2
    h2 = x[2]**2 +2*x[3]**2 -2*x[2]*x[3]*np.cos(q1-q2)
    h2 /= 2*(1 +np.sin(q1-q2)**2)**2
    p1_add = -2*np.sin(q1) -h1 +h2*np.sin(2*q1-2*q2)
    p2_add = -np.sin(q2) +h1 -h2*np.sin(2*q1-2*q2)
    p1 = x[2] +p1_add*tau
    p2 = x[3] +p2_add*tau

    return (q1, q2, p1, p2)

def eulerStep_DoublePend(x, tau, params = None):
    tau_vec = tau.reshape(tau.size(0))
    XX = torch.zeros_like(x)

    q1_add = (x[:, 0, 2] -x[:, 0, 3]*torch.cos(x[:, 0, 0]-x[:, 0, 1]))/(1 +torch.pow(torch.sin(x[:, 0, 0]-x[:, 0, 1])), 2)
    q2_add = (-x[:, 0, 2]*torch.cos(x[:, 0, 0]-x[:, 0, 1]) +2*x[:, 0, 3])/(1 +torch.pow(torch.sin(x[:, 0, 0]-x[:, 0, 1])), 2)
    q1 = x[:, 0, 0] +tau_vec*q1_add
    q2 = x[:, 0, 1] +tau_vec*q2_add

    h1 = x[:, 0, 2]*x[:, 0, 3]*torch.sin(q1-q2)
    h1 /= 1 +torch.pow(torch.sin(q1-q2), 2)
    h2 = torch.pow(x[:, 0, 2], 2) +2*torch.pow(x[:, 0, 3], 2) -2*x[:, 0, 2]*x[:, 0, 3]*torch.cos(q1-q2)
    h2 /= 2*torch.pow((1 +torch.pow(torch.sin(q1-q2), 2)), 2)
    p1_add = -2*torch.sin(q1) -h1 +h2*torch.sin(2*q1-2*q2)
    p2_add = -torch.sin(q2) +h1 -h2*torch.sin(2*q1-2*q2)
    p1 = x[:, 0, 2] +tau_vec*p1_add
    p2 = x[:, 0, 3] + tau_vec*p2_add

    XX[:, 0, 0] = q1
    XX[:, 0, 1] = q2
    XX[:, 0, 2] = p1
    XX[:, 0, 3] = p2

    return XX

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
    exact = numeric_predict_Tend(x0[i], tau, Tend, euler_stepNumpy_DoublePend, extraParams)
    exacts.append(exact)

fig, ax1 = plt.subplots(figsize=(9, 6.5))
for i in range(nr_trajects):
    ax1.plot(exacts[i][:, 0], exacts[i][:, 2])
    ax1.plot(exacts[i][0, 0], exacts[i][0, 2], marker ='o')

fig, ax2 = plt.subplots(figsize=(9, 6.5))
for i in range(nr_trajects):
    ax2.plot(exacts[i][:, 1], exacts[i][:, 3])
    ax2.plot(exacts[i][0, 1], exacts[i][0, 3], marker = 'o')