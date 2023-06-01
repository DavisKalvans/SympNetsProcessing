import numpy as np
import matplotlib.pyplot as plt
from TrainingData.harmonic_Osc import HarmOsc_H

# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

# Parameters for loading in the models predictions
N_train = 3 # Training data
M_test = 2 # Testing data
tau = 0.1
tau_txt = str(tau).replace('.', '')
epochs = 50_000
epochs_th = str(epochs/1000).replace('.0', '')
sch = True
best = False # Set to true to use the model that achieved best accuracy during training
linear = True # Set to true to use linear trained models
eta1 = 1e-3
eta2 = 1e-5 # Not used if sch is set to False
eta1_txt = str(np.log10(eta1)).replace('-', '').replace('.0', '')
eta2_txt = str(np.log10(eta2)).replace('-', '').replace('.0', '')

# Load analytical result
sol = np.load(f'TrainingData/Analytical/HarmOsc/Const{tau_txt}Tau_T1000.npy')
numeric = np.load(f'TrainingData/Numeric/HarmOsc/Euler_Const{tau_txt}Tau_T1000.npy')

Tend = 1000
M = int(Tend/tau) # Time steps
tm = np.linspace(0, Tend, M)

for nL in [2]: # Number of layers
    
    for nN in [2, 4, 8, 16, 32]: # Width of each layer
        
        for nM in [0]: # Seed
            # Load model predictions
            if sch:
                model_name = f"TrainedModels/HarmOsc/Predictions/schHarmOscRandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{eta2_txt}eta2_{nL}L{nN}n{nM}m_T1000Alternate.npy"
            else:
                model_name = f"TrainedModels/HarmOsc/Predictions/HarmOscRandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{nL}L{nN}n{nM}m_T1000Alternate.npy"
            
            if linear:
                model_name = model_name.replace('OscRand', 'OscLinearRand')

            if best:
                model_name = model_name.replace('TH', 'TH_best')

            pred = np.load(model_name)
            Mn = 1 # Number of trajectories

            Err = np.zeros([M, 1])
            HErr = np.zeros([M, 1])

            Err_numeric = np.zeros([M, 1]) # Errors for the plain numerical method
            HErr_numeric = np.zeros([M, 1]) # These should be taken out of this loop, it's not the most efficient way to recalculate these every time

            for k in range(Mn): # Go trough all the trajectories
                sol_tmp = sol.T[0:M, :]
                pred_tmp = pred[0:M, 0:2]
                numeric_tmp = numeric.T[0:M, :]
                    
                H0 = HarmOsc_H(pred_tmp[0, 0:2].reshape((1, 2)))
                Err += np.sqrt(np.sum((pred_tmp -sol_tmp)**2, 1)).reshape((M, 1))
                HErr += np.abs((HarmOsc_H(pred_tmp).reshape((M, 1)) -H0)/H0)

                Err_numeric += np.sqrt(np.sum((numeric_tmp -sol_tmp)**2, 1)).reshape((M, 1))
                HErr_numeric += np.abs((HarmOsc_H(numeric_tmp).reshape((M, 1)) -H0)/H0)


            # Plot errors
            if linear:
                plot_name = model_name.replace('Predictions/', f'Predictions/Linear/T{str(Tend)}/')
            else:
                plot_name = model_name.replace('Predictions/', f'Predictions/Sigmoid/T{str(Tend)}/')
            plot_name = plot_name.replace('_T1000Alternate.npy', '')

            # Plots global error
            fig1, ax = plt.subplots(figsize=(9, 6.5))
            ax.plot(tm, Err/Mn, ls='-', color='k', linewidth='1', label='New method') 
            ax.plot(tm, Err_numeric/Mn, ls='-', color='g', linewidth='1', label='Euler method') 
            ax.legend(loc=1, prop={'size':20})
            ax.set_xlabel("time")
            ax.set_ylabel("absolute error")
            ax.grid(True) 
            ax.set_title(f'Global error: L={str(nL)}, n={str(nN)}, m={str(nM)}')
            plt.savefig(plot_name + '_globalError', dpi=300, bbox_inches='tight')

            # Plots Hamiltonian error
            fig2, ax = plt.subplots(figsize=(9, 6.5))
            ax.plot(tm, HErr/Mn, ls='-', color='k', linewidth='1', label='New method') 
            ax.plot(tm, HErr_numeric/Mn, ls='-', color='g', linewidth='1', label='Euler method') 
            ax.legend(loc=1, prop={'size':20})
            ax.set_xlabel("time")
            ax.set_ylabel("absolute relative error")
            ax.grid(True) 
            ax.set_title(f'Hamiltonian error: L={str(nL)}, n={str(nN)}, m={str(nM)}')
            plt.savefig(plot_name + '_hamError', dpi=300, bbox_inches='tight')

            # Plots phase space of q1 and p1
            fig3, ax = plt.subplots(figsize=(9, 6.5))
            ax.grid(True)
            ax.set_xlabel("$q1$")
            ax.set_ylabel("$p1$")
            ax.set_title(f'HarmOsc: L={str(nL)}, n={str(nN)}, m={str(nM)}')

            ax.plot(sol_tmp[:, 0], sol_tmp[:, 1], ls='-', color='k', linewidth='1', label='Analytical')
            ax.plot(pred_tmp[:, 0], pred_tmp[:, 1], ls='--', color='r', linewidth='1', label='New method')
            ax.plot(numeric_tmp[:, 0], numeric_tmp[:, 1], ls='--', color='g', linewidth='1', label='Euler method')
            ax.legend(loc=1, prop={'size':20})
            plt.savefig(plot_name + '_result1', dpi=300, bbox_inches='tight')