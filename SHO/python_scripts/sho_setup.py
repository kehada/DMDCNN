import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import Hermite
import matplotlib.animation as animation

''' Figure Parameters '''
f_size = 16
figure_size = (10,8)
fileID = 3

''' Model Parameters ''' 
M = 1000 
domain = [-2,2]
x = np.linspace(domain[0],domain[1],M)
max_n = 5

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
        IC[random_index] = 1
        IC[0] = 1
        IC = IC/np.sqrt(np.sum(IC**2))
        return IC
    else:
        # return np.ones(n)/np.sqrt(n)        
        IC[0] = 1
        IC[1] = 1
        return IC


eigenvectors = np.zeros([max_n,M])
eigenvalues = np.zeros(max_n)

for i in range(0,max_n):
    eigenvectors[i] = psi_n(n=i,omega=2,x=x)
    eigenvalues[i] = E_n(n=i,omega=2)

a = generate_IC(n=max_n,random=True)

def PSI(t):
    exponentials = np.diag(np.exp(-1j*eigenvalues*t))
    return np.dot(np.dot(eigenvectors.T,exponentials),a.T)

def animate_wave(times):
    ims = []
    fig = plt.figure()
    for i in range(0,len(times)):
        im = plt.plot(np.abs(PSI(times[i])),'-k')
        ims.append(im)
    ani = animation.ArtistAnimation(fig, ims, interval=90, blit=False,repeat_delay=1000)      
    ani.save('../figures/wave_function'+str(fileID)+'.gif',dpi=100)
    plt.close()    

animate_wave(np.linspace(0,5,100))
    