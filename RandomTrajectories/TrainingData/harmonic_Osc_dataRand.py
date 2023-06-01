import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp   
from harmonic_Osc import HarmOsc

### Getting testing and training data
Tau = 0.1 # Constant time step
seed = 3 # Seed for generating data
np.random.seed(seed)

N = 100 # Ammount of training data
train_X = np.zeros((N, 1, 2))
train_Y = np.zeros((N, 1, 2))
train_Tau = Tau*np.ones((N, 1, 1))
train_q = 2.8*np.random.rand(N, 1) - 1.4
train_p = 1.6*np.random.rand(N, 1) - 0.8 


M = 40 # Ammount of testing data data
test_X = np.zeros((M, 1, 2))
test_Y = np.zeros((M, 1, 2))
test_Tau = Tau*np.ones((M, 1, 1))
test_q = 2.8*np.random.rand(M, 1) - 1.4
test_p = 1.6*np.random.rand(M, 1) - 0.8 


# Plotting options
font = {'family' : 'serif',
        'size'   : 16}
plt.rc('font', **font)

fig1, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xlabel("$q$")
ax.set_ylabel("$p$")         
ax.set_title("Phase portrait")  
ax.grid(True)
ax.axis([-1.6, 1.6, -1, 1])


# Compute training data
for n in range(N):
    train_X[n, 0, 0] = train_q[n]
    train_X[n, 0, 1] = train_p[n]    
    tau = train_Tau[n]
    # solve pendulum equations with RK45
    sol = solve_ivp(HarmOsc, [0, tau], [train_q[n].item(), train_p[n].item()], 
                    method='RK45', rtol = 1e-12, atol = 1e-12) 
    # save data
    train_Y[n, 0, 0] = sol.y[0][-1]
    train_Y[n, 0, 1] = sol.y[1][-1]

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
    sol = solve_ivp(HarmOsc, [0, tau], [test_q[m].item(), test_p[m].item()], 
                    method='RK45', rtol = 1e-12, atol = 1e-12) 
    # save data
    test_Y[m, 0, 0] = sol.y[0][-1]
    test_Y[m, 0, 1] = sol.y[1][-1]
    
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
np.savez(f'SavedTrainingData/HarmOsc/HarmOscRandN{N}M{M}ConstTau{Tau_txt}', train_X=train_X, train_Y=train_Y, train_Tau=train_Tau, test_X=test_X, test_Y=test_Y, test_Tau=test_Tau)