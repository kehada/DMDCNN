from matplotlib import rc
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]    
plt.rcParams.update({'font.size': 26})

fs = 40

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


def PSI(eigenvectors,eigenvalues,a,t):
    exponentials = np.diag(np.exp(-1j*eigenvalues*t))
    return np.dot(np.dot(eigenvectors.T,exponentials),a.T)

choice = [3,4,5]
# choice = [1,2,3]

sample_rate = 10


# snapshots=np.load('ani_snapshot.npy').view(complex)
# V=np.load('ani_V.npy')
# V_=np.load('ani_V_.npy')
# E=np.load('ani_E.npy')
# E_=np.load('ani_E_.npy')
# A = np.load('ani_A.npy').view(complex)
# A_ = np.load('ani_A_.npy').view(complex)


M = 500
Mtime = 500
domain = [-2,2]
x = np.linspace(domain[0],domain[1],M)
t = np.linspace(0,2,Mtime)

space = Mtime//10

# fig,ax = plt.subplots(5,2,figsize=(10,4))

xx,tt = np.meshgrid(x,t)

for j in choice:
    fig = plt.figure(figsize=(25,15))
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    ax2 = fig.add_subplot(2, 1, 2, projection='3d')

    ax2.grid(b=0)
    ax1.grid(b=0)

    eigenvalues = E[j,:]
    eigenvectors = V[j,:,:]

    eigenvalues_ = E_[j,:]
    eigenvectors_ = V_[j,:,:]

    a = A[j,:]
    a_ = A_[j,:]

    for i in range(0,len(t)):
        time_series = PSI(eigenvectors,eigenvalues,a,t[i])
        ax1.plot(x,np.ones(M)*t[i],np.abs(time_series),'k',linewidth=0.5,alpha=0.6)
        if not i % space :
            ax1.plot(x,np.ones(M)*t[i],np.abs(time_series),'m',linewidth=2)
        
        time_series = PSI(eigenvectors_,eigenvalues_,a_,t[i])
        ax2.plot(x,np.ones(M)*t[i],np.abs(time_series),'k',linewidth=0.5,alpha=0.6)
        if not i % space :
            ax2.plot(x,np.ones(M)*t[i],np.abs(time_series),'m',linewidth=2)    

    ax1.set_xlim([-2,2])
    ax1.set_ylim([0,2])

    ax2.set_xlim([-2,2])
    ax2.set_ylim([0,2])

    ax1.set_xlabel(r'$q$',labelpad=30,fontsize=fs)
    ax1.set_ylabel(r'$t$',labelpad=50,fontsize=fs)
    ax1.set_zlabel(r'$|\Psi^{t}(q,t)|$',labelpad=30,fontsize=fs)
    
    ax2.set_xlabel(r'$q$',labelpad=30,fontsize=fs)
    ax2.set_ylabel(r'$t$',labelpad=50,fontsize=fs)
    ax2.set_zlabel(r'$|\Psi^{m}(q,t)|$',labelpad=30,fontsize=fs)


    elev = 35

    plt.tight_layout()
    # plt.locator_params(axis='y', nbins=4)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(4))


    ax1.yaxis.set_major_locator(plt.MaxNLocator(10))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(10))


    ax1.zaxis.set_major_locator(plt.MaxNLocator(6))
    ax2.zaxis.set_major_locator(plt.MaxNLocator(6))    



    for ii in range(270,360,5):
        ax1.view_init(elev=elev, azim=ii)
        ax2.view_init(elev=elev, azim=ii)
        plt.savefig('../figures/temp/view'+str(ii)+'_'+str(j)+'.png')


    plt.close()