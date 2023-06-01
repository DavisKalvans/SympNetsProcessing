import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp   
from kepler import Kepler

### Getting testing and training data
Tau = 0.01 # Constant time step
seed = 3 # Seed for generating data
np.random.seed(seed)

N = 40 # Ammount of training data
train_X = np.zeros((N, 1, 4))
train_Y = np.zeros((N, 1, 4))
train_Tau = Tau*np.ones((N, 1, 1))
train_q1 = 3*np.random.rand(N, 1) - 1.5
train_q2 = 3*np.random.rand(N, 1) - 1.5
train_p1 = 2*np.random.rand(N, 1) - 1
train_p2 = 2*np.random.rand(N, 1) - 1 

M = 100 # Ammount of testing data data
test_X = np.zeros((M, 1, 4))
test_Y = np.zeros((M, 1, 4))
test_Tau = Tau*np.ones((M, 1, 1))
test_q1 = 3*np.random.rand(M, 1) - 1.5
test_q2 = 3*np.random.rand(M, 1) - 1.5
test_p1 = 2*np.random.rand(M, 1) - 1
test_p2 = 2*np.random.rand(M, 1) - 1


# Plotting options
font = {'family' : 'serif',
        'size'   : 16}
plt.rc('font', **font)

fig1, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xlabel("$q1$")
ax.set_ylabel("$q2$")         
ax.set_title("Phase portrait")  
ax.grid(True)


# Compute training data
for n in range(N):
    train_X[n, 0, 0] = train_q1[n]
    train_X[n, 0, 1] = train_q2[n]    
    train_X[n, 0, 2] = train_p1[n]
    train_X[n, 0, 3] = train_p2[n]
    tau = train_Tau[n]
    # solve pendulum equations with RK45
    sol = solve_ivp(Kepler, [0, tau], [train_q1[n].item(), train_q2[n].item(), train_p1[n].item(), train_p2[n].item()], 
                    method='RK45', rtol = 1e-12, atol = 1e-12) 
    # save data
    train_Y[n, 0, 0] = sol.y[0][-1]
    train_Y[n, 0, 1] = sol.y[1][-1]
    train_Y[n, 0, 2] = sol.y[2][-1]
    train_Y[n, 0, 3] = sol.y[3][-1]
    ax.plot(np.stack((train_X[n, 0, 0], train_Y[n, 0, 0])),
            np.stack((train_X[n, 0, 1], train_Y[n, 0, 1])), 
            ls='-', color='k', linewidth='1.5')
    ax.plot(train_X[n, 0, 0], train_X[n, 0, 1], marker='*', color = 'k')
    ax.plot(train_Y[n, 0, 0], train_Y[n, 0, 1], marker='o', color = 'r')

# compute testing data
for m in range(M):
    test_X[m, 0, 0] = test_q1[m]
    test_X[m, 0, 1] = test_q2[m]
    test_X[m, 0, 2] = test_p1[m]
    test_X[m, 0, 3] = test_p2[m]
    tau = test_Tau[m]
    # solve pendulum equations with RK45
    sol = solve_ivp(Kepler, [0, tau], [test_q1[m].item(), test_q2[m].item(), test_p1[m].item(), test_p2[m].item()], 
                    method='RK45', rtol = 1e-12, atol = 1e-12) 
    # save data
    test_Y[m, 0, 0] = sol.y[0][-1]
    test_Y[m, 0, 1] = sol.y[1][-1]
    test_Y[m, 0, 2] = sol.y[2][-1]
    test_Y[m, 0, 3] = sol.y[3][-1]
    
    #==========================================================================
    # plot numerical result in phase plane
    #==============================================================================   
    #ax.plot(np.stack((test_X[m, 0, 0], test_Y[m, 0, 0])),
    #        np.stack((test_X[m, 0, 1], test_Y[m, 0, 1])), 
    #        ls='-', color='k', linewidth='1.5')
    #ax.plot(test_X[m, 0, 0], test_X[m, 0, 1], marker='o', ms=7, mfc='g')
    #ax.plot(test_Y[m, 0, 0], test_Y[m, 0, 1], marker='o', ms=7, mfc='y')
# Saving data
Tau_txt = str(Tau).replace('.', '')
np.savez(f'SavedTrainingData/Kepler/KeplerRandN{N}M{M}ConstTau{Tau_txt}', train_X=train_X, train_Y=train_Y, train_Tau=train_Tau, test_X=test_X, test_Y=test_Y, test_Tau=test_Tau)