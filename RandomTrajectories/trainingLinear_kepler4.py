import numpy as np
import time
import torch
import copy
from torch import nn
from NeuralNetwork.custom_dataset import CustomDataset
from NeuralNetwork.mySequential import mySequential
from NeuralNetwork.symp_module_class import LinSympGradModule
from NeuralNetwork.training_class import train_loop_kepler, test_loop_kepler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Plotting options
import matplotlib
matplotlib.rc('font', size=24)
matplotlib.rc('axes', titlesize=20)

# Find device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print('Using {} device'.format(device))
torch.set_num_threads(1)

def grad_plotter(ave_grads, max_grads, layers):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    ax.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("average gradient")
    plt.show()


# Load training data
N = 10
M = 5
tau = 0.01
tau_txt = str(tau).replace('.', '')

data = np.load(f'TrainingData/SavedTrainingData/Kepler/KeplerRandN{N}M{M}ConstTau{tau_txt}.npz')
x_train = torch.from_numpy(np.float32(data['train_X'])).to(device)  
y_train = torch.from_numpy(np.float32(data['train_Y'])).to(device) 
tau_train = torch.from_numpy(np.float32(data['train_Tau'])).to(device) 
x_test = torch.from_numpy(np.float32(data['test_X'])).to(device) 
y_test = torch.from_numpy(np.float32(data['test_Y'])).to(device) 
tau_test = torch.from_numpy(np.float32(data['test_Tau'])).to(device) 
omega = torch.tensor(0.5, dtype=torch.float32, device=device)

# Dimension of the problem
D = x_train.shape[2]
d = int(D/2)

# Custom Dataset 
training_data = CustomDataset(x_train, y_train, tau_train)
testing_data = CustomDataset(x_test, y_test, tau_test)

# Training parameter values
learning_rate = 1e-3
batch_size = N
epochs = 50_000
epochs_th = str(epochs/1000).replace('.0', '')
sch = True
eta1 = 1e-3
eta2 = 1e-6

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

for nL in [2]:
    
    for nN in [2, 4, 8, 16, 32]:
        
        if nN == 0:
            nN += 1
        for nM in [0]:    
                
            #==================================================================
            # Symplectic neural network
            #==================================================================
            for nnet in [0]:
                    
                # Define neural network of L number of modules 
                layers = []
                for n in  range(nL):
                    layers.append(LinSympGradModule(d, nN, nL))
                model = mySequential(*layers).to(device)
                print(model)
                        
                # Set random seed to generate the same initial parameter values
                torch.manual_seed(nM)
                sigma = np.sqrt(0.01)
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
                start = time.time()
    
                # Training
                loss = np.zeros((epochs, 1))
                acc = np.zeros((epochs, 1))
                acc_best = np.inf
                h = tau/nL
                ### Debug for getting gradient of training
                #grads = []
                for t in range(epochs):
                    loss[t] = train_loop_kepler(train_dataloader, model,
                                                    loss_fn, optimizer, scheduler, sch)
                    acc[t] = test_loop_kepler(test_dataloader, model, loss_fn)

                    if acc[t] < acc_best: # Copies the model as 'best model' if it's accuracy is best yet in training
                        acc_best = acc[t]
                        model_best = copy.deepcopy(model)
                    
                    if t % 100 == 0:
                            print('Epoch %d / loss: %.12f / acc: %.12f' % (t+1, loss[t], acc[t]))

    
                # end time
                end = time.time()
                # total time taken
                print(f"Runtime of the program was {(end - start)/60:.4f} min.")


                f = str(nL) + "L" + str(nN) + "n" + str(nM) + "m"
                if sch:
                    file_w = f'TrainedModels/Kepler/schKeplerLinearRandN{N}M{M}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_{eta2_txt}eta2_' +f
                else:
                    file_w = f'TrainedModels/Kepler/KeplerLinearRandN{N}M{M}Const{tau_txt}Tau{epochs_th}TH_{eta1_txt}eta1_' +f
                
                # Save models
                model = model.to('cpu')
                torch.save([model, loss, acc, start, end], file_w)
                model_best = model_best.to('cpu')
                torch.save([model_best, loss, acc, start, end], file_w.replace('TH', 'TH_best'))

                #==================================================================
                # Plot training error vs. validation error
                #===============================================================
                fig, ax = plt.subplots(figsize=(9, 6.5))
                v = np.linspace(1, epochs, epochs)
                ax.loglog(v, loss[:, 0], ls='-', color='tab:red', linewidth='1.5', label='Loss')
                ax.loglog(v, acc, ls='--', color='tab:blue', linewidth='1.5', label='Accuracy')
                ax.set_xlabel("epochs")
                ax.set_ylabel("error")
                ax.grid(True)
                ax.legend(loc=3, shadow=True, prop={'size': 20}) 
                ax.axis([1, epochs, 10**(-9), 10^3])
                plt.yticks([10**(-11), 10**(-9), 10**(-7), 10**(-5), 10**(-3), 10**(-1), 10**1])
                ax.set_title(f'Kepler, L={nL}, N={nN}, m={nM}')
                # Save figure
                plt.savefig(file_w.replace('/Kepler/', '/Kepler/MSE/') +".png", dpi=300, bbox_inches='tight')
                #plt.show()

                del model

                ### Debug for getting gradient of training
                # print("Gradients at the start of training")
                # grad_plotter(grads[0][0], grads[0][1], grads[0][2])
                # print(f"The avarage gradients are: {grads[0][0]}")

                # print("Gradients at the end of training")
                # grad_plotter(grads[-1][0], grads[-1][1], grads[-1][2])
                # print(f"The avarage gradients are: {grads[-1][0]}")