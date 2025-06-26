
from scipy.stats import binned_statistic
from matplotlib import rc


plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]    
plt.rcParams.update({'font.size': 15})
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# MAE = np.mean(np.mean(np.abs(V-V_),axis=2),axis=1)

# bin_means, bin_edges, binnumber = binned_statistic(test_omega,
#                 MAE, statistic='mean')

# plt.figure()
# plt.plot(test_omega,MAE,'.k',markersize=1)
# plt.plot(bin_edges[:-1],bin_means,'b')
# plt.show()
# plt.close()


choice = [370,742,803]


# fig,ax = plt.subplots(2,3,figsize=(15,10))
# for i in range(0,3):
#     for j in range(0,r):
#         ax[0,i].plot(x,V_[choice[i],j,:],'k',linewidth=1)
#         ax[0,i].plot(x,V[choice[i],j,:],'--m',linewidth=1)
#         ax[1,i].plot(x,V_[choice[i],j,:]-V[choice[i],j,:],label='j='+str(j))
#         ax[1,i].set_ylim(-3.5e-2,3.5e-2)
#     ax[0,i].text(-1.5,1,'$V_0= $'+str(round(test_omega[choice[i]],3)))

# ax[1,0].legend(prop={'size': 10})
# ax[1,1].set_xlabel(r'$q$')
# ax[0,0].set_ylabel(r'$\psi_j(q)$')
# ax[1,0].set_ylabel(r'Residual')
# plt.savefig('residuals.png',dpi=300)



''' Generate new data for test '''
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import Hermite
import matplotlib.animation as animation
FFwriter = animation.FFMpegWriter()


''' Model Parameters ''' 
M = 500 
domain = [-2,2]
x = np.linspace(domain[0],domain[1],M)
max_n = 5
new_training = True
new_additional = True

''' Functions '''
def H_n(n,x):
    if n>0 :
        order_vector = np.zeros(n,dtype=int)
        order_vector[n-1] = 1        
        hermite_series = Hermite(tuple(order_vector),domain=domain)
    else : hermite_series = Hermite(1,domain=domain)
    return hermite_series(x)

def psi_n(n,omega,x):
    return 1/np.sqrt(2**n*np.math.factorial(n))*(omega/np.pi)**(1/4)*np.exp(-omega*x**2/2)*H_n(n,np.sqrt(omega)*x)

def E_n(n,omega):
    return omega/2*(2*n+1)

def generate_IC(n=max_n,random=False):
    IC = np.zeros(n)
    index = np.arange(1,n)
    if random:
        random_index = np.random.choice(index)
        IC[random_index] = np.random.rand() + 1/2
        # IC[1:] = np.random.rand(n-1)/10
        IC[0] = 1
        IC = IC/np.sqrt(np.sum(IC**2))
        return IC
    else:
        IC[0] = 1
        IC[1] = 1
        return IC

def PSI(t):
    exponentials = np.diag(np.exp(-1j*eigenvalues*t))
    return np.dot(np.dot(eigenvectors.T,exponentials),a.T)

def animate_wave(times,fileID):
    ims = []
    fig = plt.figure()
    for i in range(0,len(times)):
        im = plt.plot(np.abs(PSI(times[i])),'-k')
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=90, blit=False,repeat_delay=1000)      
    ani.save('../figures/wave_function'+fileID+'.mp4',dpi=100,writer = FFwriter)
    plt.close()    

eigenvectors = np.zeros([max_n,M])
eigenvalues = np.zeros(max_n)*1j



data_list = []
files_per_frequency = 1
Nomega = 10
mean_omega = 5
std_omega = 1
omega_values = np.linspace(2,8,Nomega)
omega_index = 0
for omega_value in omega_values:
    for i in range(0,max_n):
        eigenvectors[i] = psi_n(n=i,omega=omega_value,x=x)
        eigenvalues[i] = E_n(n=i,omega=omega_value)
    snap_time = 5*2*np.pi/mean_omega
    data_res = np.zeros([max_n+1,M+1])*0j
    a = generate_IC(n=max_n,random=True)
    snapshot = PSI(t=np.random.normal(snap_time,1))
    data_res[0,0:M]=snapshot
    data_res[0,M] = omega_value
    for i in range(0,max_n):
        data_res[1+i,0:M] = eigenvectors[i,:]
        data_res[1+i,M] = eigenvalues[i]
    omega_index+=1
    data_list.append(data_res)


