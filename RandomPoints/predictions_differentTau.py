import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp 
from TrainingData.harmonic_Osc import HarmOsc, HarmOsc_H
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)

# Parameters for loading in the models
d = 2 # Dimension of the problem
omega = 0.5
N_train = 200 # Training data
M_test = 40 # Testing data
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

# Number of steps with model
M = 81920
tau_new = 0.0001220703125
tau_new_txt = str(tau_new).replace('.', '')
Tend = tau_new*M
Tend_txt = str(Tend).replace('.0', '')
tm = np.linspace(0, Tend, M+1)

# Starting point for trajectory to predict
# Might need to load it dynamically later, if starting to use different trajectories,
# but I'll deal with that later if need be
q1 = 0.3
p1 = 0.5

# Get analytical result
sol = solve_ivp(HarmOsc, [0, Tend], [q1, p1], method = 'LSODA', t_eval = tm,
                rtol = 1e-12, atol = 1e-12)

sol = sol.y.T 

# Get Euler method result for comparison
pred_numeric = np.zeros([M+1, 2])
pred_numeric[0, :] = [q1, p1]

# Euler method
for i in range(M):
    pred_numeric[i+1, 1] = pred_numeric[i, 1] -omega**2*pred_numeric[i, 0]*tau_new
    pred_numeric[i+1, 0] = pred_numeric[i, 0] + pred_numeric[i+1, 1]*tau_new

# Errors for numeric method
H0 = HarmOsc_H(sol[0, 0:2].reshape((1, 2)))
Err_numeric = np.sqrt(np.sum((pred_numeric -sol)**2, 1)).reshape((M+1, 1))
HErr_numeric = np.abs((HarmOsc_H(pred_numeric).reshape((M+1, 1)) -H0)/H0)




for nL in [2]: # Number of layers
    
    for nN in [32]: # Width of each layer
        
        for nM in [0]: # Seed
            # Load model
            if sch:
                model_name = f"TrainedModels/HarmOsc/schHarmOscRandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{eta2_txt}eta2_{nL}L{nN}n{nM}m"
            else:
                model_name = f"TrainedModels/HarmOsc/HarmOscRandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{nL}L{nN}n{nM}m"
            
            if linear:
                model_name = model_name.replace('OscRand', 'OscLinearRand')

            if best:
                model_name = model_name.replace('TH', 'TH_best')

            model, loss, acc, start, end = torch.load(model_name)
            print(f"Runtime of training was {(end - start)/60:.4f} min.") 
            print(f"Loss was {loss[-1]} and accuracy was {acc[-1]}")

            pred_inv = np.zeros([M+1, d])
            Z = torch.tensor([[[q1, p1]]], dtype=torch.float32, device=device)
            Tau = torch.tensor([[[tau_new]]], dtype=torch.float32, device=device)
            Tau_old = torch.tensor([[[tau]]], dtype=torch.float32, device=device)

            # For the numerical method
            XX = torch.zeros((1, 1, d), dtype=torch.float32)


            with torch.no_grad():
                inverse, _ = model.back(Z, Tau) # Pass trough inverse model
                pred_inv[0, 1] = inverse[:, 0, 1]
                pred_inv[0, 0] = inverse[:, 0, 0]

                # Need to do the numerical method now (symplectic Euler)
                a = inverse[:, 0, 1] - omega**2*Tau.T*inverse[:, 0, 0]
                b = inverse[:, 0, 0] + a[0]*Tau

                a = a.reshape((1, 1))
                b = b.reshape((1, 1))
                pred_inv[1, 1] = a
                pred_inv[1, 0] = b
                

                for m in range(M-1):
                    pred_inv[m+2, 1] = pred_inv[m+1, 1] - omega**2*tau_new*pred_inv[m+1, 0]
                    pred_inv[m+2, 0] = pred_inv[m+1, 0] +pred_inv[m+2, 1]*tau_new

                pred_inv = torch.from_numpy(np.float32(pred_inv)).reshape(M+1, 1, 2)
                pred, _ = model(pred_inv, Tau) # Pass trough original model

            # Errors for neural network prediction
            pred = pred.reshape(M+1, 2)
            pred = pred.numpy()
            Err = np.sqrt(np.sum((pred-sol)**2, 1)).reshape((M+1, 1))
            HErr = np.abs((HarmOsc_H(pred).reshape((M+1, 1)) -H0)/H0)

            # Plot errors
            plot_name = model_name.replace('HarmOsc/', f'HarmOsc/Predictions_differentTau/')
            plot_name = plot_name + f'_new{tau_new_txt}Tau_T{Tend_txt}'

            # Plots global error
            fig1, ax = plt.subplots(figsize=(9, 6.5))
            ax.plot(tm, Err, ls='-', color='k', linewidth='1', label='New method') 
            ax.plot(tm, Err_numeric, ls='-', color='g', linewidth='1', label='Euler method') 
            ax.legend(loc=1, prop={'size':20})
            ax.set_xlabel("time")
            ax.set_ylabel("absolute error")
            ax.grid(True) 
            ax.set_title(f'Global error: L={str(nL)}, n={str(nN)}, m={str(nM)}')
            #plt.savefig(plot_name + '_globalError', dpi=300, bbox_inches='tight')

            # Plots Hamiltonian error
            fig2, ax = plt.subplots(figsize=(9, 6.5))
            ax.plot(tm, HErr, ls='-', color='k', linewidth='1', label='New method') 
            ax.plot(tm, HErr_numeric, ls='-', color='g', linewidth='1', label='Euler method') 
            ax.legend(loc=1, prop={'size':20})
            ax.set_xlabel("time")
            ax.set_ylabel("absolute relative error")
            ax.grid(True) 
            ax.set_title(f'Hamiltonian error: L={str(nL)}, n={str(nN)}, m={str(nM)}')
            #plt.savefig(plot_name + '_hamError', dpi=300, bbox_inches='tight')

            # Plots phase space of q1 and p1
            fig3, ax = plt.subplots(figsize=(9, 6.5))
            ax.grid(True)
            ax.set_xlabel("$q1$")
            ax.set_ylabel("$p1$")
            ax.set_title(f'HarmOsc: L={str(nL)}, n={str(nN)}, m={str(nM)}')

            ax.plot(sol[:, 0], sol[:, 1], ls='-', color='k', linewidth='1', label='Analytical')
            ax.plot(pred[:, 0], pred[:, 1], ls='--', color='r', linewidth='1', label='New method')
            ax.plot(pred_numeric[:, 0], pred_numeric[:, 1], ls='--', color='g', linewidth='1', label='Euler method')
            ax.legend(loc=1, prop={'size':20})
            #plt.savefig(plot_name + '_result1', dpi=300, bbox_inches='tight')

            # file_name = model_name.replace("TrainedModels/", f"TrainedModels/Predictions/")
            # file_name = f"{file_name}_T{Tend_txt}Alternate"
            # np.save(file_name, pred)

            
