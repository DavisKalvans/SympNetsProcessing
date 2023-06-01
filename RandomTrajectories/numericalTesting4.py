import numpy as np
from TrainingData.general_problems import Pendulum, HarmOsc, Kepler
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
import cProfile
import pstats
device = 'cpu'

'''
Energy is better with Verlet eigth order vs solve_ivp, but usually the error is only of order -11 to -13
'''
with cProfile.Profile() as pr:
    # Plotting options
    import matplotlib
    matplotlib.rc('font', size=24)
    matplotlib.rc('axes', titlesize=20)

    mse_func = torch.nn.MSELoss()
    x0_HarmOsc = (0.3, 0.5)
    x0_Pendulum = (0.8, 0.5)
    e = 0.6
    q1 = 1 - e
    q2 = 0
    p1 = 0
    p2 = np.sqrt((1+e)/(1-e))
    x0_Kepler = (q1, q2, p1, p2)

    # Triple jump coefficients (Hairier 4.4)
    def gen_gamma(p):
        gamma = np.zeros(3)
        gamma[0] = 1/(2-2**(1/(p+1)))
        gamma[1] = -(2**(1/(p+1))) /(2-2**(1/(p+1)))
        gamma[2] = 1/(2-2**(1/(p+1)))

        return gamma

    def fourth_order(x0, tau, D, gamma, problem, extraParams):
        tau_half = tau/2

        # First time
        p = x0[D:(2*D)] - gamma[0]*tau_half*problem.H_p(x0[0:D], extraParams)
        Q = x0[0:D] +gamma[0]*tau*problem.H_q(p, extraParams)
        P = p - gamma[0]*tau_half*problem.H_p(Q, extraParams)

        # Second time
        p = P - gamma[1]*tau_half*problem.H_p(Q, extraParams)
        Q = Q + gamma[1]*tau*problem.H_q(p, extraParams)
        P = p - gamma[1]*tau_half*problem.H_p(Q, extraParams)

        # Third time
        p = P - gamma[2]*tau_half*problem.H_p(Q, extraParams)
        Q = Q + gamma[2]*tau*problem.H_q(p, extraParams)
        P = p - gamma[2]*tau_half*problem.H_p(Q, extraParams)

        return np.concatenate((Q, P))

    def sixth_order(x0, tau, D, gamma2, gamma4, problem, extraParams):
        # First time
        tau1 = tau*gamma4[0]
        x1 = fourth_order(x0, tau1, D, gamma2, problem, extraParams)

        #Second time
        tau2 = tau*gamma4[1]
        x2 = fourth_order(x1, tau2, D, gamma2, problem, extraParams)

        # Third time
        tau3 = tau*gamma4[2]
        x3 = fourth_order(x2, tau3, D, gamma2, problem, extraParams)

        return x3

    def eight_order(x0, tau, D, gamma2, gamma4, gamma6, problem, extraParams):
        # First time
        tau1 = tau*gamma6[0]
        x1 = sixth_order(x0, tau1, D, gamma2, gamma4, problem, extraParams)

        tau2 = tau*gamma6[1]
        x2 = sixth_order(x1, tau2, D, gamma2, gamma4, problem, extraParams)

        tau3 = tau*gamma6[2]
        x3 = sixth_order(x2, tau3, D, gamma2, gamma4, problem, extraParams)

        return x3

    def eight_orderPrecise(x0, tau, D, gamma2, gamma4, gamma6, problem, extraParams):
        TAU = 0.01
        steps = 1

        while tau > TAU: # Constantly divide timestep by 2 and multiple required steps by 2
            tau /= 2
            steps *= 2

        for i in range(steps):
            x0 = eight_order(x0, tau, D, gamma2, gamma4, gamma6, problem, extraParams)

        return x0


    gamma2 = gen_gamma(2)
    gamma4 = gen_gamma(4)
    gamma6 = gen_gamma(6)

    ### Select problem
    problem = Pendulum
    if problem == Pendulum:
        prob_text = "Pendulum"
        extraParams = None
        x0 = x0_Pendulum
        d = 2

    elif problem == HarmOsc:
        prob_text = "HarmOsc"
        extraParams = 0.5
        x0 = x0_HarmOsc
        extraParams = 0.5
        d = 2

    elif problem == Kepler:
        prob_text = "Kepler"
        extraParams = None
        x0 = x0_Kepler
        d = 4

    D = int(d/2)

    ### Calculate energy of starting point
    H0 = problem.H(np.array([x0]).reshape(1, d), extraParams)

    Tend = 36
    tau = 9
    M = int(Tend/tau) # Time steps
    tm = np.linspace(0, Tend, M+1)

    ### Inbuilt solve to get normal trajectory
    solver_acc = solve_ivp(problem.problem, [0, Tend], x0, args = (extraParams,), method = 'RK45', rtol = 1e-12, atol = 1e-12)
    solver_acc = solver_acc.y.T

    ### Solver at timestep tau
    pred_solver = solve_ivp(problem.problem, [0, Tend], x0, args = (extraParams,), t_eval = tm, method = 'RK45', rtol = 1e-12, atol = 1e-12)
    pred_solver = pred_solver.y.T

    ### Verlet at time step tau
    pred_numeric = np.zeros([M+1, d])
    pred_numeric[0, :] = x0

    for i in range(M):
        ### Function eight order
        pred_numeric[i+1, :] = eight_orderPrecise(pred_numeric[i, :], tau, D, gamma2, gamma4, gamma6, problem, extraParams)

    ### Plot phase space
    fig1, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(solver_acc[:, 0], solver_acc[:, 1], color = 'k', linewidth = '2', label = 'Accurate traject')

    ax.plot(pred_solver[:, 0], pred_solver[:, 1], color = 'r', label = "solve_ivp")
    ax.plot(pred_numeric[:, 0], pred_numeric[:, 1], color = 'g', label = "eight order Verlet", linestyle = '--')
    ax.legend(loc=4, prop={'size':20})
    ax.grid(True)
    ax.set_xlabel('q')
    ax.set_ylabel('p')
    ax.set_title(f"{prob_text} with x0={x0}")

    ### Plot energies
    pred_solver = torch.tensor(pred_solver, dtype = torch.float64, device = device).reshape((M+1, d))
    energies_solver = problem.H(pred_solver, extraParams)
    pred_numeric = torch.tensor(pred_numeric, dtype = torch.float64, device = device).reshape((M+1, d))
    energies_pred = problem.H(pred_numeric, extraParams)

    fig2, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(tm, np.ones(M+1)*H0, color = 'k', linewidth = '2', label = 'Ground truth')
    ax.plot(tm, energies_pred, color = 'g', label = 'eight order Verlet', linestyle = '--')
    ax.plot(tm, energies_solver, color = 'r', label = 'solve_ivp')
    ax.legend(loc=4, prop={'size':20})
    ax.grid(True)
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.set_title(f"{prob_text} with x0={x0}")

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()