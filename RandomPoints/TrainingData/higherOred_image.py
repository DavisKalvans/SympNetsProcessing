import matplotlib.pyplot as plt
import numpy as np

# Plotting options
font = {'family' : 'serif',
        'size'   : 16}
plt.rc('font', **font)


def gen_gamma(p):
    gamma = np.zeros(3)
    gamma[0] = 1/(2-2**(1/(p+1)))
    gamma[1] = -(2**(1/(p+1))) /(2-2**(1/(p+1)))
    gamma[2] = 1/(2-2**(1/(p+1)))

    return gamma
### Fourth order
h = 1
y = [0, 1, 2, 3]
gammas2 = gen_gamma(2)
steps = h*gammas2
hh = [0]

for i in range(len(steps)):
    hh.append(hh[-1]+steps[i])

fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 4))
ax1.set_xticks([-1, 0, 1, 2])
ax1.set_yticks([])
ax1.plot(hh, y, marker = '.', markersize=25, linewidth=3)
ax1.vlines([0], ymin = 0, ymax = 3, linestyles='dashed', label='$h=0$', color='k')
ax1.vlines([1], ymin = 0, ymax = 3, linestyles='dashed', label='$h=01$', color='k')
#plt.tick_params(top=False, bottom=False, left=False, right=False,
#                labelleft=False, labelbottom=False)   
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(False)




### Sixth order
gammas2 = gen_gamma(2)
gammas4 = gen_gamma(4)
h = 1
y = np.arange(0, 10)
steps1 = h*gammas4
hh = [0]
for i in range(len(steps1)):
    for j in range(len(gammas2)):
        hh.append(hh[-1]+steps1[i]*gammas2[j])

#fig2, ax = plt.subplots(figsize=(9, 6.5))
ax2.plot(hh, y, marker = '.', markersize=25, linewidth=3)
ax2.vlines([0], ymin = 0, ymax = 9, linestyles='dashed', label='$h=0$', color='k')
ax2.vlines([1], ymin = 0, ymax = 9, linestyles='dashed', label='$h=01$', color='k')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(True)
ax2.spines['left'].set_visible(False)
ax2.set_xticks([-1, 0, 1, 2])
ax2.set_yticks([])

### Eight order
gammas2 = gen_gamma(2)
gammas4 = gen_gamma(4)
gammas6 = gen_gamma(6)

h = 1
y = np.arange(0, 28)
steps1 = h*gammas6
steps2 = []
for i in range(len(steps1)):
    for j in range(len(gammas4)):
        steps2.append(steps1[i]*gammas4[j])

hh = [0]
for i in range(len(steps2)):
    for j in range(len(gammas2)):
        hh.append(hh[-1]+steps2[i]*gammas2[j])


#fig3, ax = plt.subplots(figsize=(9, 6.5))
ax3.plot(hh, y, marker = '.', markersize=25, linewidth=3)
ax3.vlines([0], ymin = 0, ymax = 27, linestyles='dashed', label='$h=0$', color='k')
ax3.vlines([1], ymin = 0, ymax = 27, linestyles='dashed', label='$h=01$', color='k')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(True)
ax3.spines['left'].set_visible(False)
ax3.set_xticks([-1, 0, 1, 2])
ax3.set_yticks([])