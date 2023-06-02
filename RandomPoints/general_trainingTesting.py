import numpy as np
import time
import torch
import copy
from torch import nn
from NeuralNetwork.custom_dataset import CustomDataset
from NeuralNetwork.mySequential import mySequential
from NeuralNetwork.symp_module_class import SympGradModule
from NeuralNetwork.training_class import train_loopGeneral, test_loopGeneral
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.set_num_threads(1)

torch.set_default_dtype(torch.float64) # Yess, more default precision

""" 
problem = 'Kepler'
device = 'cpu'
data = (100, 40, 0.01)
parameters = (1e-3, 10, 10_000, False, 1e-3, 1e-5)
nL = 2
nN = 2
nM = 0
extraParams = 0.5

train_model(problem, device, data, parameters, nL, nN, nM, extraParams)

"""
# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

def train_model(problem, device, data, parameters, nL, nN, nM, extraParams = None):

    ### Load training data
    N, M, tau = data
    tau_txt = str(tau).replace('.', '')

    npz_file = np.load(f'TrainingData/SavedTrainingData/{problem}/{problem}RandN{N}M{M}ConstTau{tau_txt}.npz')

    x_train = torch.from_numpy(np.float32(npz_file['train_X'])).to(device)  
    y_train = torch.from_numpy(np.float32(npz_file['train_Y'])).to(device) 
    tau_train = torch.from_numpy(np.float32(npz_file['train_Tau'])).to(device)
    x_test = torch.from_numpy(np.float32(npz_file['test_X'])).to(device)
    y_test = torch.from_numpy(np.float32(npz_file['test_Y'])).to(device)
    tau_test = torch.from_numpy(np.float32(npz_file['test_Tau'])).to(device)

    # Set extra parameters as tensor on device for problems that need it
    if problem == 'HarmOsc':
        params = torch.tensor(extraParams, dtype=torch.float32, device=device) # Omega
    else:
        params = None

    # Dimension of the problem
    D = x_train.shape[2]
    d = int(D/2)

    # Custom Dataset 
    training_data = CustomDataset(x_train, y_train, tau_train)
    testing_data = CustomDataset(x_test, y_test, tau_test)

    ### Set up all parameters

    learning_rate, batch_size, epochs, sch, eta1, eta2 = parameters
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
        layers.append(SympGradModule(d, nN, nL))
    model = mySequential(*layers).to(device)
                        
    # Set random seed to generate the same initial parameter values
    torch.manual_seed(nM)
    sigma = np.sqrt(0.01) # Does this being sqrt(0.01) ruin everything?
    for param in model.parameters():
        param.data = sigma*torch.randn(param.shape)

    model = model.to(device)

    # Initialize the loss function
    loss_fn = nn.MSELoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                                lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
                    
    # start time
    startReal = time.time()
    
    # Training
    loss = np.zeros((epochs, 1))
    acc = np.zeros((epochs, 1))
    acc_best = np.inf
    h = tau/nL
    loss_times = np.zeros(epochs)
    acc_times = np.zeros(epochs)

    for t in range(epochs):
        start = time.time()
        loss[t] = train_loopGeneral(problem, train_dataloader, model, loss_fn, optimizer, scheduler, sch, params)
        end = time.time()
        loss_times[t] = end-start

        start = time.time()
        acc[t] = test_loopGeneral(problem, test_dataloader, model, loss_fn, params)
        end = time.time()
        acc_times[t] = end-start

        if acc[t] < acc_best: # Copies the model as 'best model' if it's accuracy is best yet in training
            acc_best = acc[t]
            model_best = copy.deepcopy(model)
                    
        if t % 100 == 0:
            print('Epoch %d / loss: %.12f / acc: %.12f' % (t+1, loss[t], acc[t]))

    # end time
    endReal = time.time()
    # total time taken
    print(f"Runtime of the program was {(endReal - startReal)/60:.4f} min.")


    f = str(nL) + "L" + str(nN) + "n" + str(nM) + "m"
    if sch:
        file_w = f'TrainedModels/{problem}/sch{problem}RandN{N}M{M}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{eta2_txt}eta2_' +f
    else:
        file_w = f'TrainedModels/{problem}/{problem}RandN{N}M{M}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_' +f


    # # Save models
    # model = model.to('cpu') # Transfer to cpu, otherwise there can be problems loading on cpu if was trained and saved with gpu
    # torch.save([model, loss, acc, start, end], file_w)
    # model_best = model_best.to('cpu')
    # torch.save([model_best, loss, acc, start, end], file_w.replace('TH', 'TH_best'))

    # ### Plot MSE
    # fig, ax = plt.subplots(figsize=(9, 6.5))
    # v = np.linspace(1, epochs, epochs)
    # ax.loglog(v, loss[:, 0], ls='-', color='tab:red', linewidth='1.5', label='Loss')
    # ax.loglog(v, acc, ls='--', color='tab:blue', linewidth='1.5', label='Accuracy')
    # ax.set_xlabel("epochs")
    # ax.set_ylabel("error")
    # ax.grid(True)
    # ax.legend(loc=3, shadow=True, prop={'size': 20}) 
    # ax.axis([1, epochs, 10**(-9), 10^3])
    # plt.yticks([10**(-11), 10**(-9), 10**(-7), 10**(-5), 10**(-3), 10**(-1), 10**1])
    # ax.set_title(f'{problem}, L={nL}, N={nN}, m={nM}')
    # # Save figure
    # plt.savefig(file_w.replace(f'/{problem}/', f'/{problem}/MSE/') +".png", dpi=300, bbox_inches='tight')
    # #plt.show()

    del model
    return loss_times, acc_times


problem = 'Kepler'
device = 'cpu'
data = (100, 40, 0.01)
parameters = (1e-3, 100, 10_000, False, 1e-3, 1e-5)
nL = 2
nN = 2
nM = 0
extraParams = 0.5


loss_times, acc_times = train_model(problem, device, data, parameters, nL, nN, nM, extraParams)

maxLoss = np.max(loss_times)
meanLoss = np.mean(loss_times)
meedianLoss = np.median(loss_times)

maxAcc = np.max(acc_times)
meanAcc = np.mean(acc_times)
meedianAcc = np.median(acc_times)

print(f'Loss func exectuvion with {parameters[2]} epochs: max={maxLoss}, mean={meanLoss}, median={meedianLoss}.')
print(f'Acc func exectuvion with {parameters[2]} epochs: max={maxAcc}, mean={meanAcc}, median={meedianAcc}.')

# Baseline - total time  1.2883 min
# Loss func exectuvion with 10000 epochs: max=0.03205990791320801, mean=0.005810226368904114, median=0.00601506233215332.
# Acc func exectuvion with 10000 epochs: max=0.017771244049072266, mean=0.0016792474746704102, median=0.0.


# Without ifs - total time 1.4200 min
# Loss func exectuvion with 10000 epochs: max=0.05730867385864258, mean=0.006434678149223327, median=0.007993578910827637.
# Acc func exectuvion with 10000 epochs: max=0.03217911720275879, mean=0.0018042664289474486, median=0.0.
