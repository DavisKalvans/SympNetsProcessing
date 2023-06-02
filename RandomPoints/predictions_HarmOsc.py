### DEPRECATED ####
# Use alternate_predictions.py instead.
# This one is slower, because continually uses the naural network instead
# of doing it once to process data for Euler method and second time to process
# Euler method's results back into original space.

import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)

# Parameters for loading in the models
d = 2 # Dimension of the problem
omega = 0.5
N_train = 10 # Training data
M_test = 5 # Testing data
tau = 0.1
tau_txt = str(tau).replace('.', '')
epochs = 10_000
epochs_th = str(epochs/1000).replace('.0', '')
sch = False
best = True # Set to true to use the model that achieved best accuracy during training
eta1 = 1e-4
eta2 = 1e-5 # Not used if sch is set to False
eta1_txt = str(np.log10(eta1)).replace('-', '').replace('.0', '')
eta2_txt = str(np.log10(eta2)).replace('-', '').replace('.0', '')

# Number of steps with model
M = 1000
Tend = tau*M
Tend_txt = str(Tend).replace('.0', '')
tm = np.linspace(0, Tend, M+1)

# Starting point for trajectory to predict
# Might need to load it dynamically later, if starting to use different trajectories,
# but I'll deal with that later if need be
q1 = 0.3
p1 = 0.5

for nL in [2]: # Number of layers
    
    for nN in [2]: # Width of each layer
        
        for nM in [0]: # Seed
            # Load model
            if sch:
                model_name = f"TrainedModels/schHarmOscLinearRandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{eta2_txt}eta2_{nL}L{nN}n{nM}m"
            else:
                model_name = f"TrainedModels/HarmOscLinearRandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{nL}L{nN}n{nM}m"
            

            if best:
                model_name = model_name.replace('TH', 'TH_best')
                
            model, loss, acc, start, end = torch.load(model_name)
            print(f"Runtime of training was {(end - start)/60:.4f} min.") 
            print(f"Loss was {loss[-1]} and accuracy was {acc[-1]}")

            pred = np.zeros([M+1, d])
            Z = torch.tensor([[[q1, p1]]], dtype=torch.float32, device=device)
            Tau = torch.tensor([[[tau]]], dtype=torch.float32, device=device)

            # For the numerical method
            XX = torch.zeros((1, 1, d), dtype=torch.float32)

            for m in range(M+1):
                pred[m, :] = Z[0, 0, :]

                with torch.no_grad():
                    inverse, _ = model.back(Z, Tau) # Pass trough inverse model

                    # Need to do the numerical method now (symplectic Euler)
                    a = inverse[:, 0, 1] - torch.matmul(omega**2*Tau.T, inverse[:, 0, 0])
                    b = inverse[:, 0, 0] + a[0,  :]*Tau.T

                    a = a.reshape((1, 1, 1))
                    XX[:, 0, 1] = a
                    XX[:, 0, 0] = b

                    Z, _ = model(XX, Tau) # Pass trough original model

            # Saving the prediction
            file_name = model_name.replace("TrainedModels/", f"TrainedModels/Predictions/")
            file_name = f"{file_name}_T{Tend_txt}"
            np.save(file_name, pred)
