from matplotlib import rc
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]    
plt.rcParams.update({'font.size': 22})
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# choice = [949,972,906]

choice = [994, 975, 921]

choice = [961,995,977]

choice = [142,6,201]

choice = [556,370,742]

color_1 = 'tab:grey'
color_2 ='m'

fs = 26

lw1 = 5
lw2 = 2

alpha1=1  
alpha2=1

ls1 = '-'
ls2 = '--'

'''
eigenvectors
'''
fig = plt.figure(figsize=(10, 10)) 
gs = gridspec.GridSpec(3, 2, width_ratios=[2, 0.2]) 
ax = []
for j in range(0,3):
    ax_ = []
    for i in range(0,2):
        ax_.append(plt.subplot(gs[j,i]))
    ax.append(ax_)
for j in range(0,r):
    ax[0][0].plot(x,V_[choice[0],j,:],color_1,linewidth=lw1,linestyle=ls1,alpha=alpha1)
    ax[0][0].plot(x,V[choice[0],j,:],color_2,linewidth=lw2,linestyle=ls2,alpha=alpha2)
    ax[0][0].text(-1.5,1,'$V_0= $'+str(round(test_omega[choice[0]],3)))

    ax[1][0].plot(x,V_[choice[1],j,:],color_1,linewidth=lw1,linestyle=ls1,alpha=alpha1)
    ax[1][0].plot(x,V[choice[1],j,:],color_2,linewidth=lw2,linestyle=ls2,alpha=alpha2)
    ax[1][0].text(-1.5,1,'$V_0= $'+str(round(test_omega[choice[1]],3)))


    ax[2][0].plot(x,V_[choice[2],j,:],color_1,linewidth=lw1,linestyle=ls1,alpha=alpha1)
    ax[2][0].plot(x,V[choice[2],j,:],color_2,linewidth=lw2,linestyle=ls2,alpha=alpha2)
    ax[2][0].text(-1.5,1,'$V_0= $'+str(round(test_omega[choice[2]],3)))


    ax[0][1].plot(np.linspace(-1,1,3),np.ones(3)*E_[choice[0],j]/test_omega[choice[0]]*2,color_1,linewidth=5)
    ax[0][1].plot(np.linspace(-1,1,3),np.ones(3)*E[choice[0],j]/test_omega[choice[0]]*2,color_2,linestyle='--',linewidth=2)

    ax[1][1].plot(np.linspace(-1,1,3),np.ones(3)*E_[choice[1],j]/test_omega[choice[1]]*2,color_1,linewidth=5)
    ax[1][1].plot(np.linspace(-1,1,3),np.ones(3)*E[choice[1],j]/test_omega[choice[1]]*2,color_2,linestyle='--',linewidth=2)

    ax[2][1].plot(np.linspace(-1,1,3),np.ones(3)*E_[choice[2],j]/test_omega[choice[2]]*2,color_1,linewidth=5)
    ax[2][1].plot(np.linspace(-1,1,3),np.ones(3)*E[choice[2],j]/test_omega[choice[2]]*2,color_2,linestyle='--',linewidth=2)        

ax[0][1].set_ylabel('$2i\omega_j/V_0$',fontsize=fs)
ax[1][1].set_ylabel('$2i\omega_j/V_0$',fontsize=fs)
ax[2][1].set_ylabel('$2i\omega_j/V_0$',fontsize=fs)
ax[2][1].set_xlabel('',fontsize=fs)

ax[0][0].tick_params(direction='in', length=3, width=1, top=1,bottom=1,left=1,right=1)
ax[1][0].tick_params(direction='in', length=3, width=1, top=1,bottom=1,left=1,right=1)
ax[2][0].tick_params(direction='in', length=3, width=1, top=1,bottom=1,left=1,right=1)


ax[0][1].tick_params(direction='in', length=3, width=1, top=1,bottom=1,left=1,right=1)
ax[1][1].tick_params(direction='in', length=3, width=1, top=1,bottom=1,left=1,right=1)
ax[2][1].tick_params(direction='in', length=3, width=1, top=1,bottom=1,left=1,right=1)



ax[2][0].set_xlabel('$q$',fontsize=fs)
ax[0][0].set_ylabel('$\\psi_j(q)$',fontsize=fs)
ax[1][0].set_ylabel('$\\psi_j(q)$',fontsize=fs)
ax[2][0].set_ylabel('$\\psi_j(q)$',fontsize=fs)
for i in range(0,3):
    ax[i][0].set_xlim([-1.7,1.7])
    ax[i][0].set_ylim([-0.5,1.3])
    if i <2:
        ax[i][0].xaxis.set_ticklabels([])
    ax[i][1].xaxis.set_visible(False)
# plt.tight_layout()
plt.subplots_adjust(hspace=0.1,wspace=0.3)
plt.savefig('../figures/model_outputs/draft_plots/test.png',dpi=300)
plt.close()


