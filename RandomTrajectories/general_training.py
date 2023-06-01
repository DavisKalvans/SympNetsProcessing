import numpy as np
import time
import torch
import copy
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from NeuralNetwork.custom_dataset import CustomDataset
from NeuralNetwork.mySequential import mySequential
from NeuralNetwork.symp_module_class import SympGradModule, LinSympGradModule
from NeuralNetwork.training_class import train_loopGeneral, test_loopGeneral
from TrainingData.general_problems import HarmOsc, Pendulum, Kepler
from TrainingData import verlet8

torch.set_num_threads(1)

# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

torch.set_default_dtype(torch.float64) # Yess, more default precision, even though I already set it up manually

### I probably need to rewrite all of these without the stupid XX and torch.cat instead - might get a speedup?
### With Kepler and 10_000 epochs nL=2, nN=2 -> 0.8465 mins; torch.cat -> xxx mins
### With Kepler and 10_000 epochs nL=8, nN=32 -> 2.5198 mins; torch.cat -> xxx mins
### Euler step functions with tensors for the training and testing loops
def eulerStep_HarmOsc(x, tau, params):
    omega = params
    tau_vec = tau.reshape(tau.size(0))
    XX = torch.zeros_like(x)


    p = x[:, 0, 1] -omega**2*tau_vec*x[:, 0, 0]
    q = x[:, 0, 0] +tau_vec*p

    XX[:, 0, 0] = q
    XX[:, 0, 1] = p

    return XX

def eulerStep_Pendulum(x, tau, params = None):
    tau_vec = tau.reshape(tau.size(0))
    XX = torch.zeros_like(x)

    p = x[:, 0, 1] -tau_vec*torch.sin(x[:, 0, 0])
    q = x[:, 0, 0] +tau_vec*p

    XX[:, 0, 0] = q
    XX[:, 0, 1] = p

    return XX

def eulerStep_Kepler(x, tau, params = None):
    tau_vec = tau.reshape(tau.size(0))
    XX = torch.zeros_like(x)

    q1 = x[:, 0, 0] +tau_vec*x[:, 0, 2]
    q2 = x[:, 0, 1] +tau_vec*x[:, 0, 3]

    d = torch.pow(torch.pow(q1, 2) +torch.pow(q2, 2), 1.5)
    p1 = x[:, 0, 2] -tau_vec*q1/d
    p2 = x[:, 0, 3] -tau_vec*q2/d

    XX[:, 0, 0] = q1
    XX[:, 0, 1] = q2
    XX[:, 0, 2] = p1
    XX[:, 0, 3] = p2

    return XX

def eulerStep_DoublePend(x, tau, params = None):
    tau_vec = tau.reshape(tau.size(0))
    XX = torch.zeros_like(x)

    q1_add = (x[:, 0, 2] -x[:, 0, 3]*torch.cos(x[:, 0, 0]-x[:, 0, 1]))/(1 +torch.pow(torch.sin(x[:, 0, 0]-x[:, 0, 1]), 2))
    q2_add = (-x[:, 0, 2]*torch.cos(x[:, 0, 0]-x[:, 0, 1]) +2*x[:, 0, 3])/(1 +torch.pow(torch.sin(x[:, 0, 0]-x[:, 0, 1]), 2))
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

### Verlet step functions with tensors for the training and testing loops
def verletStep_HarmOsc(x, tau, params):
    omega = params
    tau_vec = tau.reshape(tau.size(0))
    tau_vec_half = tau_vec/2
    XX = torch.zeros_like(x)

    p_half = x[:, 0, 1] -omega**2*tau_vec_half*x[:, 0, 0]
    q = x[:, 0, 0] +tau_vec*p_half
    p = p_half -omega**2*tau_vec_half*q

    XX[:, 0, 0] = q
    XX[:, 0, 1] = p

    return XX

def verletStep_Pendulum(x, tau, params = None):
    tau_vec = tau.reshape(tau.size(0))
    tau_vec_half = tau_vec/2
    XX = torch.zeros_like(x)

    p_half = x[:, 0, 1] -tau_vec_half*torch.sin(x[:, 0, 0])
    q = x[:, 0, 0] +tau_vec*p_half
    p = p_half -tau_vec_half*torch.sin(q)

    XX[:, 0, 0] = q
    XX[:, 0, 1] = p

    return XX

def verletStep_Kepler(x, tau, params = None):
    tau_vec = tau.reshape(tau.size(0))
    tau_vec_half = tau_vec/2
    XX = torch.zeros_like(x)

    q1_half = x[:, 0, 0] +tau_vec_half*x[:, 0, 2]
    q2_half = x[:, 0, 1] +tau_vec_half*x[:, 0, 3]

    d = torch.pow(torch.pow(q1_half, 2) +torch.pow(q2_half, 2), 1.5)
    p1 = x[:, 0, 2] -tau_vec*q1_half/d
    p2 = x[:, 0, 3] -tau_vec*q2_half/d

    q1 = q1_half +tau_vec_half*p1
    q2 = q2_half +tau_vec_half*p2

    XX[:, 0, 0] = q1
    XX[:, 0, 1] = q2
    XX[:, 0, 2] = p1
    XX[:, 0, 3] = p2

    return XX

### As much as I'd love to use only the tensor versions, they are way slower
### Euler step functions as numpy arrays
def eulerStepNumpy_HarmOsc(x, tau, params):
    omega = params
    p = x[1] -omega**2*tau*x[0]
    q = x[0] +tau*p

    return (q, p)

def eulerStepNumpy_Pendulum(x, tau, params = None):
    p = x[1] -tau*np.sin(x[0])
    q = x[0] +tau*p

    return (q, p)

def eulerStepNumpy_Kepler(x, tau, params = None):
    q1 = x[0] +tau*x[2]
    q2 = x[1] +tau*x[3]

    d = (q1**2 +q2**2)**(1.5)
    p1 = x[2] -tau*q1/d
    p2 = x[3] -tau*q2/d

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

### Verlet step functions as numpy arrays
def verletStepNumpy_HarmOsc(x, tau, params):
    omega = params
    tau_half = tau/2

    p_half = x[1] -omega**2*tau_half*x[0]
    q = x[0] +tau*p_half
    p = p_half -omega**2*tau_half*q

    return (q, p)

def verletStepNumpy_Pendulum(x, tau, params = None):
    tau_half = tau/2

    p_half = x[1] -tau_half*np.sin(x[0])
    q = x[0] +tau*p_half
    p = p_half -tau_half*np.sin(q)

    return (q, p)

def verletStepNumpy_Kepler(x, tau, params = None):
    tau_half = tau/2
    
    q1_half = x[0] +x[2]*tau_half
    q2_half = x[1] +x[3]*tau_half

    d = (q1_half**2 +q2_half**2)**(1.5)
    p1 = x[2] -q1_half/d*tau
    p2 = x[3] -q2_half/d*tau

    q1 = q1_half +p1*tau_half
    q2 = q2_half +p2*tau_half

    return (q1, q2, p1, p2)


"""  Example on how to use the training function
problem = 'Kepler'
method = 'Verlet'
device = 'cpu'
data = (10, 5, 0.1) # Training data N, testing data M, time step tau
nL = 8 # Layers
nN = 32 # Width of each layer
nM = 0 # Seed for model parameter initialization
extraParams = 0.5 # Extra parameters for problems that need it, eg, HarmOsc has omega; None by default

e = 0.6
q1 = 1 - e
q2 = 0
p1 = 0
p2 = np.sqrt((1+e)/(1-e))
x0_Kepler = (q1, q2, p1, p2)
x0_HarmOsc = (0.3, 0.5)
x0_Pendulum = (0.8, 0.5)

trainParams = (1e-3, 10, 10_000, True, 1e-3, 1e-5, False) # Learning rate, batch size, epochs, scheduling, eta1, eta2, linear
predParams = (100000, x0_Kepler, False) # Time steps, x0, best
plotParams = (1000, 2, False) # Time step to plot until, dimension, best
convParams = (x0_Kepler, 10, 3, 14, 4, False) # x0, Tend, lower power, upper power, dim, best

train_model(problem, method, device, data, trainParams, nL, nN, nM, extraParams)

model_predictions(problem, method, device, data, trainParams, nL, nN, nM, predParams, extraParams)

model_errors(problem, method, data, trainParams, nL, nN, nM, plotParams, extraParams)

model_conv(problem, method, device, data, trainParams, nL, nN, nM, convParams, extraParams)

"""

### Actual general model training function
def train_model(problem, method, device, data, trainParams, nL, nN, nM, extraParams = None):

    ### Load training data
    N, M, tau = data
    tau_txt = str(tau).replace('.', '')

    npz_file = np.load(f'TrainingData/SavedTrainingData/{problem}/{problem}RandN{N}M{M}ConstTau{tau_txt}.npz')

    x_train = torch.from_numpy(np.float64(npz_file['train_X'])).to(device)  
    y_train = torch.from_numpy(np.float64(npz_file['train_Y'])).to(device) 
    tau_train = torch.from_numpy(np.float64(npz_file['train_Tau'])).to(device)
    x_test = torch.from_numpy(np.float64(npz_file['test_X'])).to(device)
    y_test = torch.from_numpy(np.float64(npz_file['test_Y'])).to(device)
    tau_test = torch.from_numpy(np.float64(npz_file['test_Tau'])).to(device)

    # Set extra parameters as tensor on device for problems that need it
    if problem == 'HarmOsc':
        params = torch.tensor(extraParams, dtype=torch.float64, device=device) # Omega
    else:
        params = None

    # Select appropriate numeric_step function
    if method == 'Euler':
        if problem == 'HarmOsc':
            numeric_step = eulerStep_HarmOsc
        elif problem == 'Pendulum':
            numeric_step = eulerStep_Pendulum
        elif problem == 'Kepler':
            numeric_step = eulerStep_Kepler
        elif problem == "DoublePend":
            numeric_step = eulerStep_DoublePend
    elif method == "Verlet":
        if problem == 'HarmOsc':
            numeric_step = verletStep_HarmOsc
        elif problem == 'Pendulum':
            numeric_step = verletStep_Pendulum
        elif problem == 'Kepler':
            numeric_step = verletStep_Kepler 

    # Dimension of the problem
    D = x_train.shape[2]
    d = int(D/2)

    # Custom Dataset 
    training_data = CustomDataset(x_train, y_train, tau_train)
    testing_data = CustomDataset(x_test, y_test, tau_test)

    ### Set up all parameters

    learning_rate, batch_size, epochs, sch, eta1, eta2, linear = trainParams
    epochs_th = str(epochs/1000).replace('.0', '')

    gamma = np.exp(np.log(eta2/eta1)/epochs)
    if sch:
        learning_rate = eta1
        eta1_txt = str(np.log10(eta1)).replace('-', '').replace('.0', '')
        eta2_txt = str(np.log10(eta2)).replace('-', '').replace('.0', '')
    else:
        eta1_txt = str(np.log10(learning_rate)).replace('-', '').replace('.0', '')

    # Data loader for PyTorch
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size)

    ### Create the model and train

    layers = []
    for n in  range(nL):
        if linear:
            layers.append(LinSympGradModule(d, nN, nL))
        else:
            layers.append(SympGradModule(d, nN, nL))

    model = mySequential(*layers).to(device)
                        
    # Set random seed to generate the same initial parameter values
    torch.manual_seed(nM)
    sigma = np.sqrt(0.01) # Does this being sqrt(0.01) ruin everything?
    for param in model.parameters():
        param.data = sigma*torch.randn(param.shape, dtype = torch.float64)

    model = model.to(device)

    # Initialize the loss function
    loss_fn = nn.MSELoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                                lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
                    
    # start time
    start = time.time()
    
    # Training
    loss = np.zeros((epochs, 1))
    acc = np.zeros((epochs, 1))
    acc_best = np.inf
    h = tau/nL

    for t in range(epochs):
        loss[t] = train_loopGeneral(numeric_step, train_dataloader, model, loss_fn, optimizer, scheduler, sch, params)
        acc[t] = test_loopGeneral(numeric_step, test_dataloader, model, loss_fn, params)

        if acc[t] < acc_best: # Copies the model as 'best model' if it's accuracy is best yet in training
            acc_best = acc[t]
            model_best = copy.deepcopy(model)
                    
        if t % 100 == 0:
            print('Epoch %d / loss: %.16f / acc: %.16f' % (t+1, loss[t], acc[t]))

    # end time
    end = time.time()
    # total time taken
    print(f"Runtime of the program was {(end - start)/60:.4f} min.")


    f = str(nL) + "L" + str(nN) + "n" + str(nM) + "m"
    if sch:
        model_name = f'TrainedModels/{method}/{problem}/sch{problem}RandN{N}M{M}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{eta2_txt}eta2_' +f
    else:
        model_name = f'TrainedModels/{method}/{problem}/{problem}RandN{N}M{M}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_' +f

    if linear:
        model_name = model_name.replace(f'{problem}Rand', f'{problem}LinearRand')

    # Save models
    model = model.to('cpu') # Transfer to cpu, otherwise there can be problems loading on cpu if was trained and saved with gpu
    torch.save([model, loss, acc, start, end], model_name)
    model_best = model_best.to('cpu')
    torch.save([model_best, loss, acc, start, end], model_name.replace('TH', 'TH_best'))

    ### Plot MSE
    fig, ax = plt.subplots(figsize=(9, 6.5))
    v = np.linspace(1, epochs, epochs)
    ax.loglog(v, loss[:, 0], ls='-', color='tab:red', linewidth='1.5', label='Loss')
    ax.loglog(v, acc, ls='--', color='tab:blue', linewidth='1.5', label='Accuracy')
    ax.set_xlabel("epochs")
    ax.set_ylabel("error")
    ax.grid(True)
    ax.legend(loc=3, shadow=True, prop={'size': 20}) 
    if method == 'Euler':
        ax.axis([1, epochs, 10**(-9), 10^3])
        plt.yticks([10**(-11), 10**(-9), 10**(-7), 10**(-5), 10**(-3), 10**(-1), 10**1])
    elif method == 'Verlet':
        ax.axis([1, epochs, 10**(-15), 10^1])
        plt.yticks([10**(-15), 10**(-13), 10**(-11), 10**(-9), 10**(-7), 10**(-5), 10**(-3), 10**(-1), 10**1])

    ax.set_title(f'{problem}, L={nL}, N={nN}, m={nM}')
    # Save figure
    plt.savefig(model_name.replace(f'/{problem}/', f'/{problem}/MSE/') +".png", dpi=300, bbox_inches='tight')
    #plt.show()

    del model


### General function for calculating and plotting all the errors of selected model predictions
def model_errors(problem, method, device, data, trainParams, nL, nN, nM, predParams, extraParams = None):

    ### Process the provided parameters
    N_train, M_test, tau = data
    tau_txt = str(tau).replace('.', '')

    M, tau, x0, best = predParams
    d = len(x0)
    D = int(d/2)
    Tend = tau*M
    tm = np.linspace(0, Tend, M+1)
    Tend_txt = str(Tend).replace('.0', '')

    learning_rate, batch_size, epochs, sch, eta1, eta2, linear = trainParams
    epochs_th = str(epochs/1000).replace('.0', '')
    if sch:
        learning_rate = eta1
        eta1_txt = str(np.log10(eta1)).replace('-', '').replace('.0', '')
        eta2_txt = str(np.log10(eta2)).replace('-', '').replace('.0', '')
    else:
        eta1_txt = str(np.log10(learning_rate)).replace('-', '').replace('.0', '')

    # Select appropriate numeric_step function
    if method == 'Euler':
        if problem == 'HarmOsc':
            numeric_stepNumpy = eulerStepNumpy_HarmOsc
        elif problem == 'Pendulum':
            numeric_stepNumpy = eulerStepNumpy_Pendulum
        elif problem == 'Kepler':
            numeric_stepNumpy = eulerStepNumpy_Kepler
    elif method == "Verlet":
        if problem == 'HarmOsc':
            numeric_stepNumpy = verletStepNumpy_HarmOsc
        elif problem == 'Pendulum':
            numeric_stepNumpy = verletStepNumpy_Pendulum
        elif problem == 'Kepler':
            numeric_stepNumpy = verletStepNumpy_Kepler

    if problem == 'HarmOsc':
        probl_class = HarmOsc
        x_label, y_label = ('$q$', '$p$')
    elif problem == 'Pendulum':
        probl_class = Pendulum
        x_label, y_label = ('$q$', '$p$')
    elif problem == 'Kepler':
        probl_class = Kepler
        x_label, y_label = ('$q1$', '$p1$')

    # Load selected model
    if sch:
        model_name = f"TrainedModels/{method}/{problem}/sch{problem}RandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{eta2_txt}eta2_{nL}L{nN}n{nM}m"
    else:
        model_name = f"TrainedModels/{method}/{problem}/{problem}RandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{nL}L{nN}n{nM}m"
            
    if linear:
        model_name = model_name.replace(f'{problem}Rand', f'{problem}LinearRand')

    if best:
        model_name = model_name.replace('TH', 'TH_best')

    model, *_ = torch.load(model_name)

    ### Get model predictions
    pred_inv = np.zeros([M+1, d])
    Z = torch.tensor(x0, dtype=torch.float64, device=device).reshape((1, 1, d))
    Tau = torch.tensor([[[tau]]], dtype=torch.float64, device=device)

    with torch.no_grad():
        inverse, _ = model.back(Z, Tau) # Pass trough inverse model

    pred_inv[0] = inverse.reshape((1, d)).numpy()

    for i in range(M):
        pred_inv[i+1] = numeric_stepNumpy(pred_inv[i], tau, extraParams)
        
    pred_inv = torch.from_numpy(np.float64(pred_inv)).reshape((M+1, 1, d))
    with torch.no_grad():
        pred, _ = model(pred_inv, Tau) # Pass trough original model

    pred = pred.numpy().reshape(M+1, d)

    ### Get numerical method predictions
    pred_numeric = np.zeros([M+1, d])
    pred_numeric[0, :] = x0

    for i in range(M): 
        pred_numeric[i+1] = numeric_stepNumpy(pred_numeric[i], tau, extraParams)

    ### Get 'analytical' results 
    exact = np.zeros([M+1, d])
    exact[0] = x0
    for i in range(M):
        exact[i+1] = verlet8.eight_orderPrecise(exact[i], tau, D, probl_class, extraParams)

    ### Absolute and Hamiltonian errors
    Err = np.sqrt(np.sum((exact -pred)**2, 1)).reshape((M+1, 1))
    Err_numeric = np.sqrt(np.sum((exact -pred_numeric)**2, 1)).reshape((M+1, 1))

    H0 = probl_class.H(exact[0].reshape((1, d)), extraParams)
    HErr = np.abs((probl_class.H(pred, extraParams).reshape((M+1, 1)) -H0)/H0)
    HErr_numeric = np.abs((probl_class.H(pred_numeric, extraParams).reshape((M+1, 1)) -H0)/H0)
    

    ### Angular error if Kepler problem
    if problem == 'Kepler':
        L0 = probl_class.L(exact[0].reshape((1, d)), extraParams)
        LErr = np.abs((probl_class.L(pred, extraParams).reshape((M+1, 1)) -L0)/L0)
        LErr_numeric = np.abs((probl_class.L(pred_numeric, extraParams).reshape((M+1, 1)) -L0)/L0)

    Tend_txt = str(Tend).replace('.0', '')
    
    ### Plot the errors
    if linear:
        plot_name = model_name.replace(f'{method}/{problem}/', f'{method}/{problem}/Predictions/Linear/T{Tend_txt}/')
    else:
        plot_name = model_name.replace(f'{method}/{problem}/', f'{method}/{problem}/Predictions/Sigmoid/T{Tend_txt}/')

    # Plots global error
    fig1, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(tm, Err, ls='-', color='k', linewidth='1', label='New method') 
    ax.plot(tm, Err_numeric, ls='-', color='g', linewidth='1', label=f'{method} method') 
    ax.legend(loc=1, prop={'size':20})
    ax.set_xlabel("time")
    ax.set_ylabel("absolute error")
    ax.grid(True) 
    ax.set_title(f'Global error: L={str(nL)}, n={str(nN)}, m={str(nM)}')
    plt.savefig(plot_name + '_globalError', dpi=300, bbox_inches='tight')

    # Plots Hamiltonian error
    fig2, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(tm, HErr, ls='-', color='k', linewidth='1', label='New method') 
    ax.plot(tm, HErr_numeric, ls='-', color='g', linewidth='1', label=f'{method} method') 
    ax.legend(loc=1, prop={'size':20})
    ax.set_xlabel("time")
    ax.set_ylabel("absolute relative error")
    ax.grid(True) 
    ax.set_title(f'Hamiltonian error: L={str(nL)}, n={str(nN)}, m={str(nM)}')
    plt.savefig(plot_name + '_hamError', dpi=300, bbox_inches='tight')

    # Plots Angular error if it's the Kepler problem
    if problem == 'Kepler':
        fig3, ax = plt.subplots(figsize=(9, 6.5))
        ax.plot(tm, LErr, ls='-', color='k', linewidth='1', label='New method') 
        ax.plot(tm, LErr_numeric, ls='-', color='g', linewidth='1', label=f'{method} method') 
        ax.legend(loc=1, prop={'size':20})
        ax.set_xlabel("time")
        ax.set_ylabel("absolute relative error")
        ax.grid(True) 
        ax.set_title(f'Angular error: L={str(nL)}, n={str(nN)}, m={str(nM)}')
        plt.savefig(plot_name + '_angError', dpi=300, bbox_inches='tight')

    # Plots phase space of q1 and p1
    fig4, ax = plt.subplots(figsize=(9, 6.5))
    ax.grid(True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'{problem}: L={str(nL)}, n={str(nN)}, m={str(nM)}')

    ax.plot(exact[:, 0], exact[:, 1], ls='-', color='k', linewidth='1', label='Analytical')
    ax.plot(pred[:, 0], pred[:, 1], ls='--', color='r', linewidth='1', label='New method')
    ax.plot(pred_numeric[:, 0], pred_numeric[:, 1], ls='--', color='g', linewidth='1', label=f'{method} method')
    ax.legend(loc=1, prop={'size':20})
    plt.savefig(plot_name + '_result1', dpi=300, bbox_inches='tight')

### General function for making predictions with trained model
def model_multTrajectErrors(problem, method, device, data, trainParams, nL, nN, nM, predParamsMult, extraParams = None):

    ### Process the provided parameters
    N_train, M_test, tau = data
    tau_txt = str(tau).replace('.', '')

    M, tau, area, seed, nr_trajects, best = predParamsMult
    Tend = tau*M
    tm = np.linspace(0, Tend, M+1)
    Tend_txt = str(Tend).replace('.0', '')

    learning_rate, batch_size, epochs, sch, eta1, eta2, linear = trainParams
    epochs_th = str(epochs/1000).replace('.0', '')
    if sch:
        learning_rate = eta1
        eta1_txt = str(np.log10(eta1)).replace('-', '').replace('.0', '')
        eta2_txt = str(np.log10(eta2)).replace('-', '').replace('.0', '')
    else:
        eta1_txt = str(np.log10(learning_rate)).replace('-', '').replace('.0', '')

    # Set extra parameters as tensor on device for problems that need it
    if problem == 'HarmOsc':
        params = torch.tensor(extraParams, dtype=torch.float64, device=device) # Omega
    else:
        params = None

    # Select appropriate numeric_step function
    if method == 'Euler':
        if problem == 'HarmOsc':
            numeric_stepNumpy = eulerStepNumpy_HarmOsc
        elif problem == 'Pendulum':
            numeric_stepNumpy = eulerStepNumpy_Pendulum
        elif problem == 'Kepler':
            numeric_stepNumpy = eulerStepNumpy_Kepler
    elif method == "Verlet":
        if problem == 'HarmOsc':
            numeric_stepNumpy = verletStepNumpy_HarmOsc
        elif problem == 'Pendulum':
            numeric_stepNumpy = verletStepNumpy_Pendulum
        elif problem == 'Kepler':
            numeric_stepNumpy = verletStepNumpy_Kepler

    # For analytical solver and Hamiltonian, angular momentum
    if problem == 'HarmOsc':
        probl_class = HarmOsc
    elif problem == 'Pendulum':
        probl_class = Pendulum
    elif problem == 'Kepler':
        probl_class = Kepler 

    ### Load in the selected model
    if sch:
        model_name = f"TrainedModels/{method}/{problem}/sch{problem}RandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{eta2_txt}eta2_{nL}L{nN}n{nM}m"
    else:
        model_name = f"TrainedModels/{method}/{problem}/{problem}RandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{nL}L{nN}n{nM}m"
            
    if linear:
        model_name = model_name.replace(f'{problem}Rand', f'{problem}LinearRand')

    if best:
        model_name = model_name.replace('TH', 'TH_best')

    model, *_ = torch.load(model_name)

    ### Generate random starting points for trajectories
    D = len(area[0]) # Half dimension of problem
    d = 2*D # Actual dimension of problem
    x0 = []
    
    q_transform = area[0, :, 0] +(area[0, :, 1] -area[0, :, 0])
    p_transform = area[1, :, 0] +(area[1, :, 1] -area[1, :, 0])
    np.random.seed(seed)

    for i in range(nr_trajects):
        x0.append([])
        for j in range(D): # Append all q values
            x0[-1].append(q_transform[j]*np.random.rand())
        
        for j in range(D): # Append all p values
            x0[-1].append(p_transform[j]*np.random.rand())

    if problem == "Kepler":
        np.random.seed(seed)
        x0 = []

        for i in range(nr_trajects):
            x0.append([])
            e = np.random.rand()*0.8 +0.1 # in range [0.1, 0.9]
            x0[-1].append(1-e)
            x0[-1].append(0)
            x0[-1].append(0)
            p2 = np.sqrt((1+e)/(1-e))
            x0[-1].append(p2)
        
    
    ### Get model predictions from starting points x0
    predictions = []

    for i in range(nr_trajects):
        pred_inv = np.zeros([M+1, d])
        Z = torch.tensor(x0[i], dtype=torch.float64, device=device).reshape((1, 1, d))
        Tau = torch.tensor([[[tau]]], dtype=torch.float64, device=device)

        with torch.no_grad():
            inverse, _ = model.back(Z, Tau) # Pass trough inverse model

        pred_inv[0] = inverse.reshape((1, d)).numpy()

        for i in range (M):
            pred_inv[i+1] = numeric_stepNumpy(pred_inv[i], tau, extraParams)
        
        pred_inv = torch.from_numpy(np.float64(pred_inv)).reshape((M+1, 1, d))
        with torch.no_grad():
            pred, _ = model(pred_inv, Tau) # Pass trough original model

        pred = pred.numpy().reshape(M+1, d)
        predictions.append(pred)

    ### Get numerical model predictions from starting points x0
    predictions_numeric = []

    for i in range(nr_trajects):
        pred_numeric = np.zeros([M+1, d])
        pred_numeric[0, :] = x0[i]

        for i in range(M): 
            pred_numeric[i+1] = numeric_stepNumpy(pred_numeric[i], tau, extraParams)

        predictions_numeric.append(pred_numeric)

    ### Get 'analytical' results from starting points x0
    exacts = []

    for k in range(nr_trajects):
        exact = np.zeros([M+1, d])
        exact[0] = x0[k]
        for i in range(M):
            exact[i+1] = verlet8.eight_orderPrecise(exact[i], tau, D, probl_class, extraParams)

        exacts.append(exact)

    ### Absolute and Hamiltonian errors
    Err = np.zeros([M+1, 1])
    HErr = np.zeros([M+1, 1])
    Err_numeric = np.zeros([M+1, 1])
    HErr_numeric = np.zeros([M+1, 1]) 

    for k in range(nr_trajects):
        Err += np.sqrt(np.sum((predictions[k] -exacts[k])**2, 1)).reshape((M+1, 1))
        Err_numeric += np.sqrt(np.sum((predictions_numeric[k] -exacts[k])**2, 1)).reshape((M+1, 1))

        H0 = probl_class.H(exacts[k][0, :].reshape((1, d)), extraParams)
        HErr += np.abs((probl_class.H(predictions[k], extraParams).reshape((M+1, 1)) -H0)/H0)
        HErr_numeric += np.abs((probl_class.H(predictions_numeric[k], extraParams).reshape((M+1, 1)) -H0)/H0)

    ### Angular error if Kepler problem
    if problem == 'Kepler':
        LErr = np.zeros([M+1, 1])
        LErr_numeric = np.zeros([M+1, 1])

        for k in range(nr_trajects):
            L0 = probl_class.L(exacts[k][0, 0:d].reshape((1, d)), extraParams)
            LErr += np.abs((probl_class.L(predictions[k], extraParams).reshape((M+1, 1)) -L0)/L0)
            LErr_numeric += np.abs((probl_class.L(predictions_numeric[k], extraParams).reshape((M+1, 1)) -L0)/L0)

    if linear:
        plot_name = model_name.replace(f'{method}/{problem}/', f'{method}/{problem}/PredictionsMultiple/Linear/')
    else:
        plot_name = model_name.replace(f'{method}/{problem}/', f'{method}/{problem}/PredictionsMultiple/Sigmoid/')

    plot_name = plot_name +f'_seed{seed}_nr{nr_trajects}_T{Tend_txt}'

    # Plots global error
    fig1, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(tm, Err/nr_trajects, ls='-', color='r', linewidth='1', label='Apstrādes metode') 
    ax.plot(tm, Err_numeric/nr_trajects, ls='-', color='g', linewidth='1', label=f'{method} method') 
    ax.legend(loc=1, prop={'size':20})
    ax.set_xlabel("Laiks")
    ax.set_ylabel("Absolūtā kļūda")
    ax.grid(True) 
    ax.set_title(f'Globālā kļūda: L={str(nL)}, n={str(nN)}, m={str(nM)}')
    plt.savefig(plot_name + '_globalError', dpi=300, bbox_inches='tight')

    # Plots Hamiltonian error
    fig2, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(tm, HErr/nr_trajects, ls='-', color='r', linewidth='1', label='Apstrādes metode') 
    ax.plot(tm, HErr_numeric/nr_trajects, ls='-', color='g', linewidth='1', label=f'{method} method') 
    ax.legend(loc=1, prop={'size':20})
    ax.set_xlabel("Laiks")
    ax.set_ylabel("Absolūtā relatīvā kļūda")
    ax.grid(True) 
    ax.set_title(f'Hamiltona kļūda: L={str(nL)}, n={str(nN)}, m={str(nM)}')
    plt.savefig(plot_name + '_hamError', dpi=300, bbox_inches='tight')

    # Plots Angular error if it's the Kepler problem
    if problem == 'Kepler':
        fig3, ax = plt.subplots(figsize=(9, 6.5))
        ax.plot(tm, LErr/nr_trajects, ls='-', color='r', linewidth='1', label='Apstrādes metode') 
        ax.plot(tm, LErr_numeric/nr_trajects, ls='-', color='g', linewidth='1', label=f'{method} method') 
        ax.legend(loc=1, prop={'size':20})
        ax.set_xlabel("Laiks")
        ax.set_ylabel("Absolūtā relatīvā kļūda")
        ax.grid(True) 
        ax.set_title(f'Leņķiskā kļūda: L={str(nL)}, n={str(nN)}, m={str(nM)}')
        plt.savefig(plot_name + '_angError', dpi=300, bbox_inches='tight')


# General function for convergence plots of selected model
def model_conv(problem, method, device,  data, trainParams, nL, nN, nM, convParams, extraParams = None):
    ### Process the provided parameters
    N_train, M_test, tau = data
    tau_txt = str(tau).replace('.', '')

    x0, Tend, low_pow, upp_pow, d, best = convParams
    d = len(x0)
    D = int(d/2)
    powers = np.arange(low_pow, upp_pow, dtype = np.float64)
    taus = 10**(-powers)

    learning_rate, batch_size, epochs, sch, eta1, eta2, linear = trainParams
    epochs_th = str(epochs/1000).replace('.0', '')
    if sch:
        learning_rate = eta1
        eta1_txt = str(np.log10(eta1)).replace('-', '').replace('.0', '')
        eta2_txt = str(np.log10(eta2)).replace('-', '').replace('.0', '')
    else:
        eta1_txt = str(np.log10(learning_rate)).replace('-', '').replace('.0', '')

    # For analytical solver and Hamiltonian, angular momentum
    if problem == 'HarmOsc':
        probl_class = HarmOsc
    elif problem == 'Pendulum':
        probl_class = Pendulum
    elif problem == 'Kepler':
        probl_class = Kepler

    # Select appropriate numeric_step function
    if method == 'Euler':
        if problem == 'HarmOsc':
            numeric_stepNumpy = eulerStepNumpy_HarmOsc
        elif problem == 'Pendulum':
            numeric_stepNumpy = eulerStepNumpy_Pendulum
        elif problem == 'Kepler':
            numeric_stepNumpy = eulerStepNumpy_Kepler
    elif method == "Verlet":
        if problem == 'HarmOsc':
            numeric_stepNumpy = verletStepNumpy_HarmOsc
        elif problem == 'Pendulum':
            numeric_stepNumpy = verletStepNumpy_Pendulum
        elif problem == 'Kepler':
            numeric_stepNumpy = verletStepNumpy_Kepler 

    mse_func = torch.nn.MSELoss(reduction = 'sum') # Easier for error to use inbuilt tensor function for MSE

    ### Get "analytical" solution
    exact = verlet8.eight_orderPrecise(x0, Tend, D, probl_class, extraParams)
    exact = np.array(exact)
    exact = torch.tensor(exact, dtype = torch.float64, device = device).reshape((1, d))

    H0 = probl_class.H(np.array([x0]).reshape((1, d)), extraParams)
    if problem == 'Kepler':
        L0 = probl_class.L(np.array([x0]).reshape((1, d)))

    ### Load in the selected model
    if sch:
        model_name = f"TrainedModels/{method}/{problem}/sch{problem}RandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{eta2_txt}eta2_{nL}L{nN}n{nM}m"
    else:
        model_name = f"TrainedModels/{method}/{problem}/{problem}RandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{nL}L{nN}n{nM}m"
            
    if linear:
        model_name = model_name.replace(f'{problem}Rand', f'{problem}LinearRand')

    if best:
        model_name = model_name.replace('TH', 'TH_best')

    model, *_ = torch.load(model_name)


    # Save everything in lists
    predictions = []
    predictions_numeric = []
    energies_pred = []
    energies_numeric = []

    if problem == 'Kepler':
        angulars_pred = []
        angulars_numeric = []

    ### Calculate everything with differing tau values
    for tau in taus:

        M = int(Tend/tau) # Time steps
        tm = np.linspace(0, Tend, M+1)

        pred_inv = np.zeros([M+1, d])
        pred_numeric =np.zeros([M+1, d])
        Z = torch.tensor(x0, dtype=torch.float64, device=device).reshape((1, 1, d))
        Tau = torch.tensor([[[tau]]], dtype=torch.float64, device=device)


        ### Get predictions with just the numerical method
        pred_numeric[0] = x0

        for i in range(M): 
            pred_numeric[i+1] = numeric_stepNumpy(pred_numeric[i], tau, extraParams)

        predictions_numeric.append(torch.tensor(pred_numeric[-1]))
        energies_numeric.append(probl_class.H(pred_numeric, extraParams))
        if problem == 'Kepler':
            angulars_numeric.append(probl_class.L(pred_numeric, extraParams))

        ### Get predictions with new method
        with torch.no_grad():
            inverse, _ = model.back(Z, Tau)
        
        pred_inv[0] = inverse.reshape((1, d)).numpy()

        for i in range(M):
            pred_inv[i+1] = numeric_stepNumpy(pred_inv[i], tau, extraParams)

        pred_inv = torch.from_numpy(np.float64(pred_inv)).reshape(M+1, 1, d)
        with torch.no_grad():
            pred, _ = model(pred_inv, Tau)

        pred = pred.reshape((M+1, d))
        predictions.append(pred[-1])
        energies_pred.append(probl_class.H(pred, extraParams))
        if problem == 'Kepler':
            angulars_pred.append(probl_class.L(pred, extraParams))

        
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
        errors_energy[i] = max(abs((energies_pred[i] -H0)/H0))
        errors_energy_numeric[i] = max(abs((energies_numeric[i] -H0)/H0))

        if problem == 'Kepler':
            errors_ang[i] = max(abs((angulars_pred[i] -L0)/L0))
            errors_ang_numeric[i] = max(abs((angulars_numeric[i] -L0)/L0))

    ### Lines with specific order, to compare numeric method to new method
    if method == 'Euler':
        line1 = np.array(taus)
        line2 = np.array(taus)**2
    elif method == 'Verlet':
        line1 = np.array(taus)**2
        line2 = np.array(taus)**3

    ### Plot absolute errors
    fig1, ax = plt.subplots(figsize=(9, 6.5))
    ax.loglog(np.array(taus), errors**(0.5), color = 'r', linewidth = '2', label = 'New method')
    ax.loglog(np.array(taus), errors_numeric**(0.5), color = 'g', linewidth = '2', label = f'{method}')
    if method == 'Verlet':
        ax.loglog(np.array(taus), line1, color = 'k', label = "second order")
        ax.loglog(np.array(taus), line2, color = 'k', label = "third order")
    elif method == 'Euler':
        ax.loglog(np.array(taus), line1, color = 'k', label = "first order")
        ax.loglog(np.array(taus), line2, color = 'k', label = "second order")

    ax.legend(loc=4, prop={'size':20})
    ax.grid(True)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel('Aboslute error')
    ax.set_title(f"{problem} L={nL}, N={nN}, M={nM}")

    # Save plots
    if linear:
        plot_name = model_name.replace(f'TrainedModels/{method}/{problem}/', f'TrainedModels/{method}/{problem}/ConvergenceGraphs/Linear/')
    else:
        plot_name = model_name.replace(f'TrainedModels/{method}/{problem}/', f'TrainedModels/{method}/{problem}/ConvergenceGraphs/Sigmoid/')

    plt.savefig(plot_name + '_absolute', dpi=300, bbox_inches='tight')

    ### Plots hamiltonian errors
    fig2, ax = plt.subplots(figsize=(9, 6.5))
    ax.loglog(np.array(taus), errors_energy, color = 'r', linewidth = '2', label = 'New method')
    ax.loglog(np.array(taus), errors_energy_numeric, color = 'g', linewidth = '2', label = f'{method}')
    if method == 'Verlet':
        ax.loglog(np.array(taus), line1, color = 'k', label = "second order")
        ax.loglog(np.array(taus), line2, color = 'k', label = "third order")
    elif method == 'Euler':
        ax.loglog(np.array(taus), line1, color = 'k', label = "first order")
        ax.loglog(np.array(taus), line2, color = 'k', label = "second order")

    ax.legend(loc=4, prop={'size':20})
    ax.grid(True)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel('Max Hamiltonian error')
    ax.set_title(f"{problem} L={nL}, N={nN}, M={nM}")

    plt.savefig(plot_name + '_ham', dpi=300, bbox_inches='tight')

    ### Plots angular errors
    if problem == 'Kepler':
        fig3, ax = plt.subplots(figsize=(9, 6.5))
        ax.loglog(np.array(taus), errors_ang, color = 'r', linewidth = '2', label = 'New method')
        ax.loglog(np.array(taus), errors_ang_numeric, color = 'g', linewidth = '2', label = f'{method}')
        if method == 'Verlet':
            ax.loglog(np.array(taus), line1, color = 'k', label = "second order")
            ax.loglog(np.array(taus), line2, color = 'k', label = "third order")
        elif method == 'Euler':
            ax.loglog(np.array(taus), line1, color = 'k', label = "first order")
            ax.loglog(np.array(taus), line2, color = 'k', label = "second order")

        ax.legend(loc=4, prop={'size':20})
        ax.grid(True)
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel('Max angular error')
        ax.set_title(f"{problem} L={nL}, N={nN}, M={nM}")

        plt.savefig(plot_name + '_ang', dpi=300, bbox_inches='tight')


# General function for convergence plots of selected model
def model_multTrajectConv(problem, method, device,  data, trainParams, nL, nN, nM, convParamsMult, extraParams = None):
    ### Process the provided parameters
    N_train, M_test, tau = data
    tau_txt = str(tau).replace('.', '')

    Tend, low_pow, upp_pow, d, area, seed, nr_trajects, best = convParamsMult
    Tend_txt = str(Tend).replace('.0', '')
    powers = np.arange(low_pow, upp_pow, dtype = np.float64)
    taus = 10**(-powers)

    # Override taus
    #taus = (1, 0.8, 0.5, 0.4, 0.25, 0.2, 0.16, 0.1, 0.08, 0.05, 0.02, 0.016, 0.01)

    learning_rate, batch_size, epochs, sch, eta1, eta2, linear = trainParams
    epochs_th = str(epochs/1000).replace('.0', '')
    if sch:
        learning_rate = eta1
        eta1_txt = str(np.log10(eta1)).replace('-', '').replace('.0', '')
        eta2_txt = str(np.log10(eta2)).replace('-', '').replace('.0', '')
    else:
        eta1_txt = str(np.log10(learning_rate)).replace('-', '').replace('.0', '')

    # For analytical solver and Hamiltonian, angular momentum
    if problem == 'HarmOsc':
        probl_class = HarmOsc
    elif problem == 'Pendulum':
        probl_class = Pendulum
    elif problem == 'Kepler':
        probl_class = Kepler

    # Select appropriate numeric_step function
    if method == 'Euler':
        if problem == 'HarmOsc':
            numeric_stepNumpy = eulerStepNumpy_HarmOsc
        elif problem == 'Pendulum':
            numeric_stepNumpy = eulerStepNumpy_Pendulum
        elif problem == 'Kepler':
            numeric_stepNumpy = eulerStepNumpy_Kepler
    elif method == "Verlet":
        if problem == 'HarmOsc':
            numeric_stepNumpy = verletStepNumpy_HarmOsc
        elif problem == 'Pendulum':
            numeric_stepNumpy = verletStepNumpy_Pendulum
        elif problem == 'Kepler':
            numeric_stepNumpy = verletStepNumpy_Kepler 

    mse_func = torch.nn.MSELoss(reduction = 'sum') # Easier for error to use inbuilt tensor function for MSE

    ### Generate random starting points for trajectories
    D = len(area[0]) # Half dimension of problem
    x0 = []
    
    q_transform = area[0, :, 0] +(area[0, :, 1] -area[0, :, 0])
    p_transform = area[1, :, 0] +(area[1, :, 1] -area[1, :, 0])
    np.random.seed(seed)

    for i in range(nr_trajects):
        x0.append([])
        for j in range(D): # Append all q values
            x0[-1].append(q_transform[j]*np.random.rand())
        
        for j in range(D): # Append all p values
            x0[-1].append(p_transform[j]*np.random.rand())

    if problem == "Kepler":
        np.random.seed(seed)
        x0 = []

        for i in range(nr_trajects):
            x0.append([])
            e = np.random.rand()*0.8 +0.1 # in range [0.1, 0.9]
            x0[-1].append(1-e)
            x0[-1].append(0)
            x0[-1].append(0)
            p2 = np.sqrt((1+e)/(1-e))
            x0[-1].append(p2)

    ### Get "analytical" solutions
    exacts = []
    H0s = []
    L0s = []

    for i in range(nr_trajects):
        exact = verlet8.eight_orderPrecise(x0[i], Tend, D, probl_class, extraParams)
        exact = np.array(exact)
        exact = torch.tensor(exact, dtype = torch.float64, device = device).reshape((1, d))
        exacts.append(exact)

        H0 = probl_class.H(np.array([x0[i]]).reshape((1, d)), extraParams)
        H0s.append(H0)
        if problem == 'Kepler':
            L0 = probl_class.L(np.array([x0[i]]).reshape((1, d)))
            L0s.append(L0)

    ### Load in the selected model
    if sch:
        model_name = f"TrainedModels/{method}/{problem}/sch{problem}RandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{eta2_txt}eta2_{nL}L{nN}n{nM}m"
    else:
        model_name = f"TrainedModels/{method}/{problem}/{problem}RandN{N_train}M{M_test}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{nL}L{nN}n{nM}m"
            
    if linear:
        model_name = model_name.replace(f'{problem}Rand', f'{problem}LinearRand')

    if best:
        model_name = model_name.replace('TH', 'TH_best')

    model, *_ = torch.load(model_name)

    # To save all the values
    errors = np.zeros(len(taus))
    errors_numeric = np.zeros(len(taus))
    errors_energy = np.zeros(len(taus))
    errors_energy_numeric = np.zeros(len(taus))
    errors_ang = np.zeros(len(taus))
    errors_ang_numeric = np.zeros(len(taus))

    for k in range(nr_trajects): # Loop trough all trajectories
        # Save everything in lists
        predictions = []
        predictions_numeric = []
        energies_pred = []
        energies_numeric = []

        if problem == 'Kepler':
            angulars_pred = []
            angulars_numeric = []

        ### Calculate everything with differing tau values
        for tau in taus:

            M = int(Tend/tau) # Time steps
            tm = np.linspace(0, Tend, M)

            pred_inv = np.zeros([M+1, d])
            pred_numeric = np.zeros([M+1, d])
            Z = torch.tensor(x0[k], dtype=torch.float64, device=device).reshape((1, 1, d))
            Tau = torch.tensor([[[tau]]], dtype=torch.float64, device=device)


            ### Get predictions with just the numerical method
            pred_numeric[0, :] = x0[k]

            for i in range(M): 
                pred_numeric[i+1] = numeric_stepNumpy(pred_numeric[i], tau, extraParams)

            predictions_numeric.append(torch.tensor(pred_numeric[-1]))
            energies_numeric.append(probl_class.H(pred_numeric, extraParams))
            if problem == 'Kepler':
                angulars_numeric.append(probl_class.L(pred_numeric, extraParams))

            ### Get predictions with new method
            with torch.no_grad():
                inverse, _ = model.back(Z, Tau)
            
            pred_inv[0] = inverse.reshape((1, d)).numpy()

            for i in range(M):
                pred_inv[i+1] = numeric_stepNumpy(pred_inv[i], tau, extraParams)

            pred_inv = torch.from_numpy(np.float64(pred_inv)).reshape(M+1, 1, d)
            with torch.no_grad():
                pred, _ = model(pred_inv, Tau)

            pred = pred.reshape((M+1, d))
            predictions.append(pred[-1])
            energies_pred.append(probl_class.H(pred, extraParams))
            if problem == 'Kepler':
                angulars_pred.append(probl_class.L(pred, extraParams))

            
        ### Calculate errors for our predictions at endpoint
        # and energy/angular errors as max deviation in the whole interval [0, Tend]
        

        for i in range(len(taus)):
            errors[i] += torch.pow(mse_func(predictions[i], exacts[k]), 0.5)
            errors_numeric[i] += torch.pow(mse_func(predictions_numeric[i].reshape((1, d)), exacts[k]), 0.5)
            errors_energy[i] += max(abs((energies_pred[i] -H0s[k])/H0s[k]))
            errors_energy_numeric[i] += max(abs((energies_numeric[i] -H0s[k])/H0s[k]))

            if problem == 'Kepler':
                errors_ang[i] += max(abs((angulars_pred[i] -L0s[k])/L0s[k]))
                errors_ang_numeric[i] += max(abs((angulars_numeric[i] -L0s[k])/L0s[k]))

    ### Lines with specific order, to compare numeric method to new method
    if method == 'Euler':
        line1 = np.array(taus)
        line2 = np.array(taus)**2
    elif method == 'Verlet':
        line1 = np.array(taus)**2
        line2 = np.array(taus)**3

    ### Plot absolute errors
    fig1, ax = plt.subplots(figsize=(9, 6.5))
    ax.loglog(np.array(taus), errors/nr_trajects, color = 'r', linewidth = '2', label = 'Apstrādes metode')
    ax.loglog(np.array(taus), errors_numeric/nr_trajects, color = 'g', linewidth = '2', label = f'{method}')
    if method == 'Verlet':
        ax.loglog(np.array(taus), line1, color = 'k', label = "otrā kārta")
        ax.loglog(np.array(taus), line2, color = 'k', label = "trešā kārta")
    elif method == 'Euler':
        ax.loglog(np.array(taus), line1, color = 'k', label = "pirmā kārta")
        ax.loglog(np.array(taus), line2, color = 'k', label = "otrā kārta")

    ax.legend(loc=4, prop={'size':20})
    ax.grid(True)
    ax.set_xlabel(r'$h$')
    ax.set_ylabel('Absolūtā kļūda')
    ax.set_title(f"{problem} L={nL}, N={nN}, M={nM}")

    # Save plots
    if linear:
        plot_name = model_name.replace(f'TrainedModels/{method}/{problem}/', f'TrainedModels/{method}/{problem}/ConvergenceGraphsMultiple/Linear/')
    else:
        plot_name = model_name.replace(f'TrainedModels/{method}/{problem}/', f'TrainedModels/{method}/{problem}/ConvergenceGraphsMultiple/Sigmoid/')

    plot_name = plot_name +f'_T{Tend_txt}'
    plt.savefig(plot_name + '_absolute', dpi=300, bbox_inches='tight')

    ### Plots hamiltonian errors
    fig2, ax = plt.subplots(figsize=(9, 6.5))
    ax.loglog(np.array(taus), (errors_energy/nr_trajects), color = 'r', linewidth = '2', label = 'Apstrādes metode')
    ax.loglog(np.array(taus), (errors_energy_numeric/nr_trajects), color = 'g', linewidth = '2', label = f'{method}')
    if method == 'Verlet':
        ax.loglog(np.array(taus), line1, color = 'k', label = "otrā kārta")
        ax.loglog(np.array(taus), line2, color = 'k', label = "trešā kārta")
    elif method == 'Euler':
        ax.loglog(np.array(taus), line1, color = 'k', label = "pirmā kārta")
        ax.loglog(np.array(taus), line2, color = 'k', label = "otrā kārta")

    ax.legend(loc=4, prop={'size':20})
    ax.grid(True)
    ax.set_xlabel(r'$h$')
    ax.set_ylabel('Lielākā Hamiltona relatīvā kļūda')
    ax.set_title(f"{problem} L={nL}, N={nN}, M={nM}")

    plt.savefig(plot_name + '_ham', dpi=300, bbox_inches='tight')

    ### Plots angular errors
    if problem == 'Kepler':
        fig3, ax = plt.subplots(figsize=(9, 6.5))
        ax.loglog(np.array(taus), (errors_ang/nr_trajects), color = 'r', linewidth = '2', label = 'Apstrādes metode')
        ax.loglog(np.array(taus), (errors_ang_numeric/nr_trajects), color = 'g', linewidth = '2', label = f'{method}')
        if method == 'Verlet':
            ax.loglog(np.array(taus), line1, color = 'k', label = "otrā kārta")
            ax.loglog(np.array(taus), line2, color = 'k', label = "trešā kārta")
        elif method == 'Euler':
            ax.loglog(np.array(taus), line1, color = 'k', label = "pirmā kārta")
            ax.loglog(np.array(taus), line2, color = 'k', label = "otrā kārta")

        ax.legend(loc=4, prop={'size':20})
        ax.grid(True)
        ax.set_xlabel(r'$\h')
        ax.set_ylabel('Lielākā leņķiskā relatīvā kļūda')
        ax.set_title(f"{problem} L={nL}, N={nN}, M={nM}")

        plt.savefig(plot_name + '_ang', dpi=300, bbox_inches='tight')
