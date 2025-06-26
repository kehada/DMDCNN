import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import Hermite
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] =  'C:\\Users\\Keir\\ffmpeg\\bin\\ffmpeg.exe'
FFwriter = animation.FFMpegWriter()

def PSI(eigenvectors,eigenvalues,a,t):
    exponentials = np.diag(np.exp(-1j*eigenvalues*t))
    return np.dot(np.dot(eigenvectors.T,exponentials),a.T)

def animate_wave(times,omega_index,classifier,fileID):
    ims = []
    fig = plt.figure(figsize=(6,6))
    plt.xlabel('$x$')
    plt.ylabel('$|\Psi(x,t)|$')
    # plt.title('$\omega=$'+str(round(test_omega[omega_index],4)))
    plt.xlim([-2,2])
    for i in range(0,len(times)):
        im = plt.plot(x,np.abs(PSI(eigenvectors,eigenvalues,a,times[i])),'m',x,np.abs(PSI(eigenvectors_,eigenvalues_,a_,times[i])),'k')
        ims.append(im)        
    ani = animation.ArtistAnimation(fig, ims, interval=60, blit=False,repeat_delay=1000)      
    ani.save('../figures/animations/animation'+classifier+str(fileID)+'.mp4',writer = FFwriter,dpi=100)
    plt.close()    


M = 500
r = 5

Ntest = 250
domain = [-2,2]
x = np.linspace(domain[0],domain[1],M)
t = np.linspace(0,2,100)



snapshots=np.load('ani_snapshot.npy').view(complex)
V=np.load('ani_V.npy')
V_=np.load('ani_V_.npy')
E=np.load('ani_E.npy')
E_=np.load('ani_E_.npy')
A = np.load('ani_A.npy').view(complex)
A_ = np.load('ani_A_.npy').view(complex)

for j in range(999,980,-1):
    eigenvalues = E[j,:]
    eigenvectors = V[j,:,:]

    eigenvalues_ = E_[j,:]
    eigenvectors_ = V_[j,:,:]

    a = A[j,:]
    a_ = A_[j,:]
    animate_wave(t,j,'sho',j)


