import numpy as np
import matplotlib.pyplot as plt
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)

# Parameters for loading in the models
d = 4 # Dimension of the problem
N_train = 40 # Training data
M_test = 15 # Testing data
tau = 0.01
tau_txt = str(tau).replace('.', '')
epochs = 50_000
epochs_th = str(epochs/1000).replace('.0', '')
sch = False
best = False # Set to true to use the model that achieved best accuracy during training
linear = False # Set to true to use linear trained models
eta1 = 1e-3
eta2 = 1e-5 # Not used if sch is set to False
eta1_txt = str(np.log10(eta1)).replace('-', '').replace('.0', '')
eta2_txt = str(np.log10(eta2)).replace('-', '').replace('.0', '')

# Number of steps with model
M = 100_000
Tend = tau*M
Tend_txt = str(Tend).replace('.0', '')
tm = np.linspace(0, Tend, M+1)

# Starting point for trajectory to predict
# Might need to load it dynamically later, if starting to use different trajectories,
# but I'll deal with that later if need be
e = 0.6
q1 = 1 - e
q2 = 0
p1 = 0
p2 = np.sqrt((1+e)/(1-e))

for nL in [2, 4, 8]: # Number of layers
    
    for nN in [4, 8, 16, 32]: # Width of each layer
        
        for nM in [0]: # Seed
            # Load model
            if sch:
                model_name = f"TrainedModels/Kepler/schKeplerRandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{eta2_txt}eta2_{nL}L{nN}n{nM}m"
            else:
                model_name = f"TrainedModels/Kepler/KeplerRandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{nL}L{nN}n{nM}m"
            
            if linear:
                model_name = model_name.replace('KeplerRand', 'KeplerLinearRand')

            if best:
                model_name = model_name.replace('TH', 'TH_best')

            model, loss, acc, start, end = torch.load(model_name)
            print(f"Runtime of training was {(end - start)/60:.4f} min.") 
            print(f"Loss was {loss[-1]} and accuracy was {acc[-1]}")

            pred_inv = np.zeros([M+1, d])
            Z = torch.tensor([[[q1, q2, p1, p2]]], dtype=torch.float32, device=device)
            Tau = torch.tensor([[[tau]]], dtype=torch.float32, device=device)

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

            #Saving the prediction
            pred = pred.reshape(M+1, d)

            file_name = model_name.replace("TrainedModels/Kepler/", f"TrainedModels/Kepler/Predictions/")
                
            file_name = f"{file_name}_T{Tend_txt}Alternate"
            np.save(file_name, pred)

            plt.plot(pred[:, 0], pred[:, 1])
