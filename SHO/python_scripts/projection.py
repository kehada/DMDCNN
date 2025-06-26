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
M = 100
domain = [-2,2]
x = np.linspace(domain[0],domain[1],M)
max_n = 10

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
        # IC[random_index] = np.random.rand() + 1/2
        # IC[1:] = np.random.rand(n-1)/4
        # IC[0] = 1
        IC = np.random.rand(n)
        IC = IC/np.sqrt(np.sum(IC**2))
        return IC
    else:
        IC[0] = 1
        IC[1] = 1
        return IC

def PSI(t):
    exponentials = np.diag(np.exp(-1j*eigenvalues*t))
    return np.dot(np.dot(eigenvectors.T,exponentials),a.T)

omega_value = 5
eigenvectors = np.zeros([max_n,M])
eigenvalues = np.zeros(max_n)*1j
for i in range(0,max_n):
    eigenvectors[i] = psi_n(n=i,omega=omega_value,x=x)
    eigenvalues[i] = E_n(n=i,omega=omega_value)
    a = generate_IC(n=max_n,random=False)
    snapshot = PSI(0)


print(np.dot(np.linalg.pinv(eigenvectors.T),snapshot).real)
print(a)