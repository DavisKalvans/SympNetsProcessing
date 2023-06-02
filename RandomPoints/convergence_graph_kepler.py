import numpy as np
import matplotlib.pyplot as plt
from TrainingData.kepler import Kepler, Kepler_H, Kepler_L
from scipy.integrate import solve_ivp
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)

# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)


# Parameters for loading in the models
d = 4 # Dimension of the problem
N_train = 100 # Training data
M_test = 40 # Testing data
tau = 0.1
tau_txt = str(tau).replace('.', '')
epochs = 50_000
epochs_th = str(epochs/1000).replace('.0', '')
sch = False
best = False # Set to true to use the model that achieved best accuracy during training
linear = True # Set to true to use linear trained models
eta1 = 1e-3
eta2 = 1e-5 # Not used if sch is set to False
eta1_txt = str(np.log10(eta1)).replace('-', '').replace('.0', '')
eta2_txt = str(np.log10(eta2)).replace('-', '').replace('.0', '')


Tend = 10
a = np.arange(3, 14, dtype = np.float32)
taus = 2**(-a) # Different time steps for convergence graph

# Starting point for trajectory to predict
# Might need to load it dynamically later, if starting to use different trajectories,
# but I'll deal with that later if need be
e = 0.6
q1 = 1 - e
q2 = 0
p1 = 0
p2 = np.sqrt((1+e)/(1-e))

mse_func = torch.nn.MSELoss() # Easier for error to use inbuilt tensor function for MSE

### Get "analytical" solution

exact = solve_ivp(Kepler, [0, Tend], [q1, q2, p1, p2], method = 'RK45', rtol = 1e-12, atol = 1e-12)
exact = exact.y.T[-1] # Only want the solution at time t=Tend
exact = np.array(exact)
exact = torch.tensor(exact, dtype = torch.float32, device = device).reshape((1, d))

H0 = Kepler_H(np.array([q1, q2, p1, p2]).reshape((1, d)))
L0 = Kepler_L(np.array([q1, q2, p1, p2]).reshape((1, d)))

for nL in [2, 4, 8]: # Number of layers
    
    for nN in [4, 8, 16, 32]: # Width of each layer
        
        for nM in [0]: # Seed
            # Load models predictions
            if sch:
                model_name = f"TrainedModels/Kepler/schKeplerRandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{eta2_txt}eta2_{nL}L{nN}n{nM}m"
            else:
                model_name = f"TrainedModels/Kepler/KeplerRandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{nL}L{nN}n{nM}m"
            
            if linear:
                model_name = model_name.replace('KeplerRand', 'KeplerLinearRand')

            if best:
                model_name = model_name.replace('TH', 'TH_best')

            model, loss, acc, start, end = torch.load(model_name)

            predictions = []
            predictions_numeric = []
            energies_pred = []
            energies_numeric = []
            angulars_pred = []
            angulars_numeric = []

            for tau in taus:
                
                M = int(Tend/tau) # Time steps
                tm = np.linspace(0, Tend, M)

                pred_inv = np.zeros([M+1, d])
                pred_numeric = np.zeros([M+1, d])
                Z = torch.tensor([[[q1, q2, p1, p2]]], dtype=torch.float32, device=device)
                Tau = torch.tensor([[[tau]]], dtype=torch.float32, device=device)

                pred_numeric[0, :] = [q1, q2, p1, p2]

                # Euler method
                for i in range(M):
                    pred_numeric[i+1, 0] = pred_numeric[i, 0] +pred_numeric[i, 2]*tau
                    pred_numeric[i+1, 1] = pred_numeric[i, 1] +pred_numeric[i, 3]*tau

                    dd = (pred_numeric[i+1, 0]**2 +pred_numeric[i+1, 1]**2)**(1.5)
                    pred_numeric[i+1, 2] = pred_numeric[i, 2] -pred_numeric[i+1, 0]/dd*tau
                    pred_numeric[i+1, 3] = pred_numeric[i, 3] -pred_numeric[i+1, 1]/dd*tau

                predictions_numeric.append(torch.tensor(pred_numeric[-1]))
                energy_numeric = Kepler_H(pred_numeric)
                energies_numeric.append(energy_numeric)
                ang_numeric = Kepler_L(pred_numeric)
                angulars_numeric.append(ang_numeric)
                
                ### Get prediction from our method
                # For the numerical method
                XX = torch.zeros((1, 1, d), dtype=torch.float32)
                

                with torch.no_grad():
                    inverse, _ = model.back(Z, Tau) # Pass trough inverse model
                    pred_inv[0, 0] = inverse[:, 0, 0]
                    pred_inv[0, 1] = inverse[:, 0, 1]
                    pred_inv[0, 2] = inverse[:, 0, 2]
                    pred_inv[0, 3] = inverse[:, 0, 3]
                    

                    for m in range(M):
                        pred_inv[m+1, 0] = pred_inv[m, 0] +pred_inv[m, 2]*tau
                        pred_inv[m+1, 1] = pred_inv[m, 1] +pred_inv[m, 3]*tau

                        dd = (pred_inv[m+1, 0]**2 +pred_inv[m+1, 1]**2)**(1.5)
                        pred_inv[m+1, 2] = pred_inv[m, 2] -pred_inv[m+1, 0]/dd*tau
                        pred_inv[m+1, 3] = pred_inv[m, 3] -pred_inv[m+1, 1]/dd*tau

                    pred_inv = torch.from_numpy(np.float32(pred_inv)).reshape(M+1, 1, d)
                    pred, _ = model(pred_inv, Tau) # Pass trough original model
                    predictions.append(pred[-1])

                    energy_pred = Kepler_H(pred.reshape((M+1, d)))
                    energies_pred.append(energy_pred)
                    ang_pred = Kepler_L(pred.reshape((M+1, d)))
                    angulars_pred.append(ang_pred)

            ### Calculate errors for our predictions at endpoint
            # and energy/angular errors as max deviation in the whole interval [0, Tend]
            errors = np.zeros(len(taus))
            errors_numeric = np.zeros(len(taus))
            errors_energy = np.zeros(len(taus))
            errors_energy_numeric = np.zeros(len(taus))
            errors_ang = np.zeros(len(taus))
            errors_ang_numeric = np.zeros(len(taus))


            for i in range(len(taus)):
                errors[i] = mse_func(predictions[i], exact)
                errors_numeric[i] = mse_func(predictions_numeric[i].reshape((1, d)), exact)
                errors_energy[i] = max(abs(energies_pred[i] -H0))
                errors_energy_numeric[i] = max(abs(energies_numeric[i] -H0))
                errors_ang[i] = max(abs(angulars_pred[i] -L0))
                errors_ang_numeric[i] = max(abs(angulars_numeric[i] -L0))

            line1 = np.array(taus)
            line2 = np.array(taus)**2
            line3 = np.array(taus)**3
            line4 = np.array(taus)**4

            # Plots evabsolute errors
            fig1, ax = plt.subplots(figsize=(9, 6.5))
            ax.loglog(np.array(taus), errors**(0.5), color = 'r', linewidth = '2', label = 'New method')
            ax.loglog(np.array(taus), errors_numeric**(0.5), color = 'g', linewidth = '2', label = 'Euler')
            ax.loglog(np.array(taus), line1, color = 'k', label = "first order")
            ax.loglog(np.array(taus), line2, color = 'k', label = "second order")
            #ax.loglog(np.array(taus), line3, color = 'k', label = "third order")
            #ax.loglog(np.array(taus), line4, color = 'k', label = "fourth order")
            ax.legend(loc=4, prop={'size':20})
            ax.grid(True)
            ax.set_xlabel(r'$\tau$')
            ax.set_ylabel('Aboslute error')
            ax.set_title(f"Kepler L={nL}, N={nN}, M={nM}")

            # Save plots
            if linear:
                plot_name = model_name.replace('TrainedModels/Kepler/', f'TrainedModels/Kepler/ConvergenceGraphs/Linear/')
            else:
                plot_name = model_name.replace('TrainedModels/Kepler/', f'TrainedModels/Kepler/ConvergenceGraphs/Sigmoid/')
            plot_name = plot_name.replace('_T1000Alternate.npy', '')

            plt.savefig(plot_name + '_absolute', dpi=300, bbox_inches='tight')

            # Plots hamiltonian errors
            fig2, ax = plt.subplots(figsize=(9, 6.5))
            ax.loglog(np.array(taus), errors_energy, color = 'r', linewidth = '2', label = 'New method')
            ax.loglog(np.array(taus), errors_energy_numeric, color = 'g', linewidth = '2', label = 'Euler')
            ax.loglog(np.array(taus), line1, color = 'k', label = "first order")
            ax.loglog(np.array(taus), line2, color = 'k', label = "second order")
            ax.legend(loc=4, prop={'size':20})
            ax.grid(True)
            ax.set_xlabel(r'$\tau$')
            ax.set_ylabel('Max Hamiltonian error')
            ax.set_title(f"Kepler L={nL}, N={nN}, M={nM}")

            plt.savefig(plot_name + '_ham', dpi=300, bbox_inches='tight')

            # Plots angular errors
            fig3, ax = plt.subplots(figsize=(9, 6.5))
            ax.loglog(np.array(taus), errors_ang, color = 'r', linewidth = '2', label = 'New method')
            ax.loglog(np.array(taus), errors_ang_numeric, color = 'g', linewidth = '2', label = 'Euler')
            ax.loglog(np.array(taus), line1, color = 'k', label = "first order")
            ax.loglog(np.array(taus), line2, color = 'k', label = "second order")
            ax.legend(loc=4, prop={'size':20})
            ax.grid(True)
            ax.set_xlabel(r'$\tau$')
            ax.set_ylabel('Max angular error')
            ax.set_title(f"Kepler L={nL}, N={nN}, M={nM}")

            plt.savefig(plot_name + '_ang', dpi=300, bbox_inches='tight')



            

