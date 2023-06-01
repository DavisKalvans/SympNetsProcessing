import numpy as np
import matplotlib.pyplot as plt
from TrainingData.harmonic_Osc import HarmOsc, HarmOsc_H
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
d = 2 # Dimension of the problem
omega = 0.5
N_M_train= [[3, 2], [10, 5], [40, 20], [200, 40]] # Specify models to graph; 1st number - training data amount, 2nd - testing data amount
N_train = 10 # Training data
M_test = 5 # Testing data
tau = 0.1
tau_txt = str(tau).replace('.', '')
epochs = 50_000
epochs_th = str(epochs/1000).replace('.0', '')
sch = True
best = False # Set to true to use the model that achieved best accuracy during training
linear = False # Set to true to use linear trained models
eta1 = 1e-3
eta2 = 1e-5 # Not used if sch is set to False
eta1_txt = str(np.log10(eta1)).replace('-', '').replace('.0', '')
eta2_txt = str(np.log10(eta2)).replace('-', '').replace('.0', '')

Tend = 100 # How many steps for predictions
new_tau = 0.01 # Time step fro predictions, can differ from trained models tau
M = int(Tend/new_tau)
tm = np.arange(0, Tend+new_tau, new_tau)

# Starting point for trajectory to predict
# Might need to load it dynamically later, if starting to use different trajectories,
# but I'll deal with that later if need be
q1 = 0.3
p1 = 0.5

mse_func = torch.nn.MSELoss() # Easier for error to use inbuilt tensor function for MSE
numb_models = len(N_M_train) # How many models will be plotted

### Get "analytical" solution
exact = solve_ivp(HarmOsc, [0, Tend], [q1, p1], t_eval = tm, method = 'RK45', rtol = 1e-12, atol = 1e-12)
exact = exact.y.T
exact = np.array(exact)
exact = torch.tensor(exact, dtype = torch.float32, device = device).reshape((M+1, 1, d))

H0 = HarmOsc_H(exact[0, 0, 0:2].reshape((1, 2)))
H0 = H0.numpy()
exact = exact.numpy()


for nL in [2]: # Number of layers
    
    for nN in [2, 32]: # Width of each layer
        
        for nM in [0]: # Seed

            predictions = [] # Stores predictions of models
            errors = []
            h_errors = []

            for data in N_M_train:
                N_train, M_test = data

            # Load models
                if sch:
                    model_name = f"TrainedModels/HarmOsc/schHarmOscRandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{eta2_txt}eta2_{nL}L{nN}n{nM}m"
                else:
                    model_name = f"TrainedModels/HarmOsc/HarmOscRandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{nL}L{nN}n{nM}m"
                
                if linear:
                    model_name = model_name.replace('OscRand', 'OscLinearRand')

                if best:
                    model_name = model_name.replace('TH', 'TH_best')

                model, loss, acc, start, end = torch.load(model_name)


                pred_inv = np.zeros([M+1, d])
                pred_numeric = np.zeros([M+1, d])
                Z = torch.tensor([[[q1, p1]]], dtype=torch.float32, device=device)
                Tau = torch.tensor([[[new_tau]]], dtype=torch.float32, device=device)

                pred_numeric[0, :] = [q1, p1]
                    
                ### Get prediction from our method
                # For the numerical method
                XX = torch.zeros((1, 1, d), dtype=torch.float32)            

                with torch.no_grad():
                    inverse, _ = model.back(Z, Tau) # Pass trough inverse model
                    pred_inv[0, 1] = inverse[:, 0, 1]
                    pred_inv[0, 0] = inverse[:, 0, 0]

                    # Need to do the numerical method now (symplectic Euler)
                    a = inverse[:, 0, 1] -omega**2*Tau.mT*inverse[:, 0, 0]
                    b = inverse[:, 0, 0] + a[0]*Tau

                    a = a.reshape((1, 1))
                    b = b.reshape((1, 1))
                    pred_inv[1, 1] = a
                    pred_inv[1, 0] = b
                        

                    for m in range(M-1):
                        pred_inv[m+2, 1] = pred_inv[m+1, 1] - omega**2*new_tau*pred_inv[m+1, 0]
                        pred_inv[m+2, 0] = pred_inv[m+1, 0] +pred_inv[m+2, 1]*new_tau

                    pred_inv = torch.from_numpy(np.float32(pred_inv)).reshape(M+1, 1, 2)
                    pred, _ = model(pred_inv, Tau) # Pass trough original model
                    predictions.append(pred.numpy())
                    err = np.sqrt(np.sum((pred.numpy()-exact)**2, 2)).reshape((M+1, 1))
                    errors.append(err)
                    h_err = np.abs((HarmOsc_H(pred.reshape((M+1, 2))) -H0)/H0)
                    h_errors.append(h_err)

            ### Plot absolute errors
            fig1, ax = plt.subplots(figsize=(9, 6.5))

            for i in range(numb_models):
                label = f'{str(N_M_train[i][0])} training data'
                ax.plot(tm, errors[i][:, 0], label=label)

            ax.grid(True)
            ax.set_xlabel("time")
            ax.set_ylabel("Absolute error")
            ax.legend(loc=0, prop={'size':20})
            ax.set_title(f"Global, L={nL}, N={nN}, M={nM}, tau={new_tau}")
                

            ## Plot Hamiltonian errors

            fig2, ax = plt.subplots(figsize=(9, 6.5))

            for i in range(numb_models):
                label = f'{str(N_M_train[i][0])} training data'
                ax.plot(tm, h_errors[i], label=label)

            ax.grid(True)
            ax.set_xlabel("time")
            ax.set_ylabel("Absolute relative error")
            ax.legend(loc=0, prop={'size':20})
            ax.set_title(f"Hamilt, L={nL}, N={nN}, M={nM}, tau={new_tau}")

