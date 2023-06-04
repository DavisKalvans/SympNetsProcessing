from analysis_generalfunc import loss_acc_average, errors_average, analytical_long, kernel_long, pred_long, vpt_calc
import numpy as np
import os


### General parameters for model and problem
method = 'Euler'
problem = 'Kepler'
extraParams = 'None'
if method == "Verlet": # symetric in the sense that h^2 is used (k=2)
    sym = True
else:
    sym = False

sch = False # Model with scheduling
linear = False # Model with linear activation fucntions
best = False # Model with lowest accuraccy during training (validation error)
epochs = 1_000 # Only use thousands here
N_train = 40 # Training data
M_test = 100 # Testing data
tau_model = 0.01 # Time that was used for training
tau = 0.01 # Time step to use for predictions
learning_rate = 1e-2
eta1 = 1e-1 # For scheduling (starting learning rate)
eta2 = 1e-3 # For scheduling (ending learning rate)
nL = [2] # List of layer counts used
nN = [4] # List of layer width used
nM = np.arange(0, 2) # Amount of models for each layer and width combination

### Parameters for error predictions with multiple trajects
area_Pendulum = np.array([[[-1.5, 2]], [[-1.5, 1.5]]]) # Same as area from training data
area_HarmOsc = np.array([[[-1.4, 1.4]], [[-0.8, 0.8]]]) # Same as area from training data
area_Kepler = np.array([[[-1.5, 1.5], [-1.5, 1.5]], [[-1, 1], [-1, 1]]]) # Same as area from training data
if problem == "Pendulum":
    area = area_Pendulum
elif problem == "Kepler":
    area = area_Kepler

seed = 0
nr_trajects = 10 # How many trajectories to make predictions for
if problem == 'Kepler': # Time of predictions
    Tend = 100
elif problem == "Pendulum":
    Tend = 10


#################################################
### Loss and acc averages and standart deviations
#################################################
load_only_model = False
tau_txt = str(tau).replace('.', '')
epochs_th = str(epochs/1000).replace('.0', '')
name = f'Analysis/MSE_Acc/{method}{problem}RandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH'
if best:
    name = name +'_best'

loss_acc_params = method, problem, sym, sch, linear, best, epochs, N_train, M_test, tau_model, learning_rate, eta1, eta2, nL, nN, nM, load_only_model
#model, loss, acc = load_model(method, problem, sym, sch, linear, best, epochs, N_train, M_test, tau, learning_rate, eta1, eta2, nL, nN, nM, load_only_model)
a = loss_acc_average(loss_acc_params)

dataframe = a[4]
print("Table for average loss and accuracy for models")
print(dataframe)
dataframe.to_csv(name+'.csv', index = False)


dataframe_latex = dataframe.to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.3E}".format,
)
print(dataframe_latex)
with open(name +'.txt', 'w') as f:
    f.write(dataframe_latex)

###################
# Prediction errors
###################
load_only_model = True
loss_acc_params = method, problem, sym, sch, linear, best, epochs, N_train, M_test, tau_model, learning_rate, eta1, eta2, nL, nN, nM, load_only_model
error_params = area, nr_trajects, seed, Tend, extraParams
b = errors_average(loss_acc_params, error_params, tau)
print("Table for kenrel method's average prediction errors")
print(b[0])
print("Table for processing method's average prediction errors")
print(b[1])

#Save everything
tau_txt = str(tau).replace('.', '')
epochs_th = str(epochs/1000).replace('.0', '')
name = f'Analysis/Errors/{method}{problem}RandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_Seed{seed}nrTrajects{nr_trajects}_Tend{int(Tend)}'
if best:
    name = name +'_best'

errors_latex = b[1].to_latex(index=False,
                   formatters={"name": str.upper},
                   float_format="{:.3E}".format,
)
with open(name +'.txt', 'w') as f:
    f.write(errors_latex)


errors_individual = []
for i in range(len(nL)*len(nN)):
    errors_individual.append(b[2][i].to_latex(index=False,
                   formatters={"name": str.upper},
                   float_format="{:.3E}".format,
))

with open(name + '_ALL.txt', 'w') as f:
    for err in errors_individual:
        f.write(err)

name_kernel = f'Analysis/Errors/{method}{problem}Const{tau_txt}Tau_Seed{seed}nrTrajects{nr_trajects}_Tend{int(Tend)}'
kernel_latex = b[0].to_latex(index=False,
                   formatters={"name": str.upper},
                   float_format="{:.3E}".format,
)
with open(name_kernel +'.txt', 'w') as f:
    f.write(kernel_latex)


############################
### VPT for one model (best)
############################
if problem == "Kepler":
    Tend_vpt = 100
elif problem == "Pendulum":
    Tend_vpt = 2000

treshold = 0.1
tau = 0.05
tau_txt = str(tau).replace('.', '')
tau_model_txt = str(tau_model).replace('.', '')
load_only_model = True
# Select specific model class
nL = 2
nN = 4
#nM = 1
nM = [0]
loss_acc_params = method, problem, sym, sch, linear, best, epochs, N_train, M_test, tau_model, learning_rate, eta1, eta2, nL, nN, nM, load_only_model
error_params = area, nr_trajects, seed, Tend, extraParams

# Always check if long predictions are already calculated, otherwise calculate them
analytical_name = f'Analysis/VPT/{problem}Const{tau_txt}Tau_Seed{seed}nrTrajects{nr_trajects}_Tend{int(Tend_vpt)}_analytical.npz'
if not os.path.isfile(analytical_name):
    analytical_long(loss_acc_params, error_params, Tend_vpt, tau)

kernel_name = f'Analysis/VPT/{problem}Const{tau_txt}Tau_Seed{seed}nrTrajects{nr_trajects}_Tend{int(Tend_vpt)}_{method}.npz'
if not os.path.isfile(kernel_name):
    kernel_long(loss_acc_params, error_params, Tend_vpt, tau)

epochs_th = str(epochs/1000).replace('.0', '')
if sch:
    learning_rate = eta1
    eta1_txt = str(np.log10(eta1)).replace('-', '').replace('.0', '')
    eta2_txt = str(np.log10(eta2)).replace('-', '').replace('.0', '')
else:
    eta1_txt = str(np.log10(learning_rate)).replace('-', '').replace('.0', '')

name = f'Analysis/VPT/{problem}Const{tau_txt}Tau_Seed{seed}nrTrajects{nr_trajects}_Tend{int(Tend_vpt)}'
if sch:
    name += '_sch'
    
name += f'_Const{tau_model_txt}Tau{epochs_th}TH'
if best:
    name += '_best'

if sch:
    name += f'_{eta1_txt}eta1_{eta2_txt}eta2'
else:
    name += f'_{eta1_txt}eta1'


name += f'_{nL}L{nN}n.npz'

if not os.path.isfile(name):
    pred_long(loss_acc_params, error_params, Tend_vpt, tau)

# Load them all up
analytical = np.load(analytical_name)
analytical = analytical['arr_0']
kernel = np.load(kernel_name)
kernel = kernel['arr_0']
model_pred = np.load(name)
model_pred = model_pred['arr_0']

c = vpt_calc(analytical, kernel, model_pred, nM, tau, Tend_vpt, treshold, nr_trajects, method)
print('Average VPT table for kernel and processing methods')
print(c[2])

# Use this for getting LaTeX code for the VPT table
#c[2].to_latex(index=False,
#                   formatters={"name": str.upper},
#                   float_format="{:.2f}".format)

