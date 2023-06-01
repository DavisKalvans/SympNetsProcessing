import numpy as np
import matplotlib.pyplot as plt 
from general_problems import Kepler 
import verlet8

### Getting testing and training data
Tau = 0.1 # Constant time step
seed = 3 # Seed for generating data
seed = 9001 # Seed for generating data, big so later predictions won't use it (safeguarding)
np.random.seed(seed)
D = 2 # Half the dimension
extraParams = None

N = 320 # Ammount of training data
train_X = np.zeros((N, 1, 4))
train_Y = np.zeros((N, 1, 4))
train_Tau = Tau*np.ones((N, 1, 1))

# Starting trajectory
e = 0.6
q1 = 1 - e
q2 = 0
p1 = 0
p2 = np.sqrt((1+e)/(1-e))


M = 100 # Ammount of testing data data
e = np.random.rand(M)*0.8 +0.1
test_X = np.zeros((M, 1, 4))
test_Y = np.zeros((M, 1, 4))
test_Tau = Tau*np.ones((M, 1, 1))


# Plotting options
font = {'family' : 'serif',
        'size'   : 16}
plt.rc('font', **font)
fig1, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xlabel("$q1$")
ax.set_ylabel("$q2$")         
ax.set_title("Trenēšanas datu kopa")  
ax.grid(True)
#ax.axis([-1.6, 1.6, -1, 1])
# Compute training data

exacts = np.zeros([N+1, 4])
exacts[0] = [q1, q2, p1, p2]

for i in range(N):
    # solve pendulum equations with RK45
    tau = train_Tau[i, 0, 0]
    exacts[i+1] = verlet8.eight_orderPrecise(exacts[i], tau, D, Kepler, extraParams)
    train_X[i] = exacts[i].reshape((1, 4))
    train_Y[i] = exacts[i+1].reshape((1, 4))

    ax.plot(np.stack((train_X[i, 0, 0], train_Y[i, 0, 0])),
            np.stack((train_X[i, 0, 1], train_Y[i, 0, 1])), 
            ls='-', color='k', linewidth='1.5')
    ax.plot(train_X[i, 0, 0], train_X[i, 0, 1], marker='o', ms=7, mfc='b', mec = 'b')
    ax.plot(train_Y[i, 0, 0], train_Y[i, 0, 1], marker='o', ms=7, mfc='b', mec = 'b')


exacts2 = np.zeros([M+1, 4])
exacts2[0] = exacts[-1].reshape((1, 4))
# compute testing data
for i in range(M):
    tau = test_Tau[i, 0, 0]
    exacts2[i+1] = verlet8.eight_orderPrecise(exacts2[i], tau, D, Kepler, extraParams)

    test_X[i] = exacts2[i].reshape((1, 4))
    test_Y[i] = exacts2[i+1].reshape((1, 4))


    #==========================================================================
    # plot numerical result in phase plane
    #==============================================================================   
    #ax.plot(np.stack((test_X[i, 0, 0], test_Y[i, 0, 0])),
    #        np.stack((test_X[i, 0, 1], test_Y[i, 0, 1])), 
    #        ls='-', color='k', linewidth='1.5')
    #ax.plot(test_X[i, 0, 0], test_X[i, 0, 1], marker='*', ms=7, mfc='r', mec = 'r')
    #ax.plot(test_Y[i, 0, 0], test_Y[i, 0, 1], marker='*', ms=7, mfc='r', mec = 'r')
# Saving data
Tau_txt = str(Tau).replace('.', '')
np.savez(f'SavedTrainingData/Kepler/KeplerRandN{N}M{M}ConstTau{Tau_txt}', train_X=train_X, train_Y=train_Y, train_Tau=train_Tau, test_X=test_X, test_Y=test_Y, test_Tau=test_Tau)