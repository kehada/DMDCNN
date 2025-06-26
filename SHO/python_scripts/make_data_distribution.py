import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import Hermite
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] =  'C:\\Users\\Keir\\ffmpeg\\bin\\ffmpeg.exe'
FFwriter = animation.FFMpegWriter()

''' Figure Parameters '''
f_size = 16
figure_size = (10,8)

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

if new_training:
    files_per_frequency = 10
    Nomega = 2000
    mean_omega = 5
    std_omega = 1
    omega_values = np.random.normal(mean_omega,std_omega,Nomega)
    omega_index = 0 
    for omega_value in omega_values:
        for i in range(0,max_n):
            eigenvectors[i] = psi_n(n=i,omega=omega_value,x=x)
            eigenvalues[i] = E_n(n=i,omega=omega_value)
        for k in range(0,files_per_frequency):
            snap_time = 5*2*np.pi/mean_omega
            data = np.zeros([max_n+1,M+1])*0j
            a = generate_IC(n=max_n,random=True)
            fileID = str(omega_index)+str(k)
            # animate_wave(np.linspace(snap_time-snap_time/5,snap_time+snap_time/5,100),fileID)
            snapshot = PSI(t=np.random.normal(snap_time,1))
            data[0,0:M]=snapshot
            data[0,M] = omega_value
            for i in range(0,max_n):
                data[1+i,0:M] = eigenvectors[i,:]
                data[1+i,M] = eigenvalues[i]
            np.save('../data/omega_distribution/June22/sho_'+str(omega_index)+'_'+str(k)+'.npy',data.view(float))
        omega_index+=1


