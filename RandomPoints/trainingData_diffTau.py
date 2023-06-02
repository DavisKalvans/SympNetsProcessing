import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp   
from TrainingData.general_problems import Pendulum, HarmOsc, Kepler

"""
problem = 'Pendulum'
dataParams = (40, 20, 0.1, 0.001, 0.1) # Testing data, training data, tau (unused), min timestep, max timestep

generate_dataDiffTau(problem, dataParams, True)
"""


def generate_dataDiffTau(problem, dataParams, plot = False):
    N, M, tau, tau_min, tau_max = dataParams # tau theoretically doesn't get used, maybe remove later

    seed = 3 # Seed for generating data
    np.random.seed(seed)

    min_txt = str(tau_min).replace('.', '')
    max_txt = str(tau_max).replace('.', '')

    if plot:
        # Plotting options
        font = {'family' : 'serif',
                'size'   : 16}
        plt.rc('font', **font)

        fig1, ax = plt.subplots(figsize=(9, 6.5))      
        ax.set_title("Phase portrait")  
        ax.grid(True)


############## HarmOsc
    if problem == 'HarmOsc': ### NEED TO FINISH IMPLEMENTING THIS
        probl_class = HarmOsc
        dim = 2


############## Pendulum
    elif problem == 'Pendulum':
        probl_class = Pendulum
        dim = 2

        train_X = np.zeros((N, 1, dim))
        train_Y = np.zeros((N, 1, dim))
        train_Tau = tau_min +(tau_max-tau_min)*np.random.rand(N, 1, 1) # Generates timesteps from the interval (tau_min, tau_max)
        train_q = 3.5*np.random.rand(N, 1) - 1.5
        train_p = 3*np.random.rand(N, 1) - 1.5 

        test_X = np.zeros((M, 1, dim))
        test_Y = np.zeros((M, 1, dim))
        test_Tau = tau_min +(tau_max-tau_min)*np.random.rand(M, 1, 1) # Generates timesteps from the interval (tau_min, tau_max)
        test_q = 3.5*np.random.rand(M, 1) - 1.5
        test_p = 3*np.random.rand(M, 1) - 1.5 

        # Compute training data
        for n in range(N):
            train_X[n, 0, 0] = train_q[n]
            train_X[n, 0, 1] = train_p[n]    
            tau = train_Tau[n]
            # Solve pendulum equations with RK45
            sol = solve_ivp(probl_class.problem, [0, tau], [train_q[n].item(), train_p[n].item()], 
                            method='RK45', rtol = 1e-12, atol = 1e-12) 
            # Save data
            train_Y[n, 0, 0] = sol.y[0][-1]
            train_Y[n, 0, 1] = sol.y[1][-1]

            if plot:
                ax.set_xlabel("$q$")
                ax.set_ylabel("$p$")   

                ax.plot(np.stack((train_X[n, 0, 0], train_Y[n, 0, 0])),
                        np.stack((train_X[n, 0, 1], train_Y[n, 0, 1])), 
                        ls='-', color='k', linewidth='1.5')
                ax.plot(train_X[n, 0, 0], train_X[n, 0, 1], marker='o', ms=7, mfc='b')
                ax.plot(train_Y[n, 0, 0], train_Y[n, 0, 1], marker='o', ms=7, mfc='r')


        # Compute testing data
        for m in range(M):
            test_X[m, 0, 0] = test_q[m]
            test_X[m, 0, 1] = test_p[m]    
            tau = test_Tau[m]
            # solve pendulum equations with RK45
            sol = solve_ivp(probl_class.problem, [0, tau], [test_q[m].item(), test_p[m].item()], 
                                method='RK45', rtol = 1e-12, atol = 1e-12) 
            # save data
            test_Y[m, 0, 0] = sol.y[0][-1]
            test_Y[m, 0, 1] = sol.y[1][-1]
                
            if plot:  
                ax.plot(np.stack((test_X[m, 0, 0], test_Y[m, 0, 0])),
                    np.stack((test_X[m, 0, 1], test_Y[m, 0, 1])), 
                    ls='-', color='k', linewidth='1.5')
                ax.plot(test_X[m, 0, 0], test_X[m, 0, 1], marker='o', ms=7, mfc='g')
                ax.plot(test_Y[m, 0, 0], test_Y[m, 0, 1], marker='o', ms=7, mfc='y')

        # Saving data
        
        np.savez(f'TrainingData/SavedTrainingData/{problem}/{problem}RandN{N}M{M}tauMin{min_txt}tauMax{max_txt}',train_X=train_X, train_Y=train_Y, train_Tau=train_Tau, test_X=test_X, test_Y=test_Y, test_Tau=test_Tau)

############## Kepler
    elif problem == 'Kepler': ### NEED TO FINISH IMPLEMENTING THIS
        probl_class = Kepler
        dim = 4
