import numpy as np
import matplotlib.pyplot as plt 
import verlet8
from general_problems import Pendulum

### Getting testing and training data
Tau = 0.1 # Constant time step
seed = 3 # Seed for generating data
np.random.seed(seed)
D = 1 # Half the dimension
extraParams = None

N = 320 # Ammount of training data
train_X = np.zeros((N, 1, 2))
train_Y = np.zeros((N, 1, 2))
train_Tau = Tau*np.ones((N, 1, 1))
train_q = 3.5*np.random.rand(N, 1) - 1.5
train_p = 3*np.random.rand(N, 1) - 1.5 


M = 100 # Ammount of testing data data
test_X = np.zeros((M, 1, 2))
test_Y = np.zeros((M, 1, 2))
test_Tau = Tau*np.ones((M, 1, 1))
test_q = 3.5*np.random.rand(M, 1) - 1.5
test_p = 3*np.random.rand(M, 1) - 1.5 


# Plotting options
font = {'family' : 'serif',
        'size'   : 16}
plt.rc('font', **font)

fig1, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xlabel("$q$")
ax.set_ylabel("$p$")         
ax.set_title("Phase portrait")  
ax.grid(True)
#ax.axis([-1.6, 1.6, -1, 1])


# Compute training data
for n in range(N):
    train_X[n, 0, 0] = train_q[n]
    train_X[n, 0, 1] = train_p[n]    
    tau = train_Tau[n]

    # solve pendulum equations with RK45
    sol = verlet8.eight_orderPrecise(train_X[n, 0, :], tau, D, Pendulum, extraParams)

    # save data
    train_Y[n] = sol.reshape((1, 2))

    ax.plot(np.stack((train_X[n, 0, 0], train_Y[n, 0, 0])),
            np.stack((train_X[n, 0, 1], train_Y[n, 0, 1])), 
            ls='-', color='k', linewidth='1.5')
    ax.plot(train_X[n, 0, 0], train_X[n, 0, 1], marker='o', ms=7, mfc='b')
    ax.plot(train_Y[n, 0, 0], train_Y[n, 0, 1], marker='o', ms=7, mfc='r')


# compute testing data
for m in range(M):
    test_X[m, 0, 0] = test_q[m]
    test_X[m, 0, 1] = test_p[m]    
    tau = test_Tau[m]
    # solve pendulum equations with RK45
    sol = verlet8.eight_orderPrecise(test_X[m, 0, :], tau, D, Pendulum, extraParams)
    # save data
    test_Y[m] = sol.reshape((1, 2))
    
    #==========================================================================
    # plot numerical result in phase plane
    #==============================================================================   
    ax.plot(np.stack((test_X[m, 0, 0], test_Y[m, 0, 0])),
            np.stack((test_X[m, 0, 1], test_Y[m, 0, 1])), 
            ls='-', color='k', linewidth='1.5')
    ax.plot(test_X[m, 0, 0], test_X[m, 0, 1], marker='o', ms=7, mfc='g')
    ax.plot(test_Y[m, 0, 0], test_Y[m, 0, 1], marker='o', ms=7, mfc='y')

# Saving data
Tau_txt = str(Tau).replace('.', '')
np.savez(f'SavedTrainingData/Pendulum/PendulumRandN{N}M{M}ConstTau{Tau_txt}', train_X=train_X, train_Y=train_Y, train_Tau=train_Tau, test_X=test_X, test_Y=test_Y, test_Tau=test_Tau)