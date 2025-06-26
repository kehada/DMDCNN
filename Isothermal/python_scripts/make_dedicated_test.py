import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]    
plt.rcParams.update({'font.size': 16})
import pickle
import sys
from DMDresources import DMD_model
from DMDresources import dataContainer
plt.rcParams['animation.ffmpeg_path'] =  'C:\\Users\\Keir\\ffmpeg\\bin\\ffmpeg.exe'
FFwriter = animation.FFMpegWriter()

# Units: Assume G = 1
one_kpc = 1.
one_pc = 0.001
one_kms = 0.01
one_Msun = 1./2.3225e9

# Physical parameters for system
SurfaceDensity = 60.*one_Msun/(one_pc**2.)
vsig = 20.*one_kms
z0 = (vsig**2)/np.pi/SurfaceDensity
rho0 = SurfaceDensity/2/z0

# Simulation parameters
''' previously had good results with N = 100000 '''
N = 100000
particle_SD = SurfaceDensity/np.float(N)
zbox = 8.*z0
wbox = 8.*vsig
boxrange = ((-zbox,zbox),(-wbox,wbox))

orbital_frequency = np.sqrt(2.)*vsig/z0
orbital_period = 2.*np.pi/orbital_frequency
''' previously had good results with dt = orbital_period/500'''
dt = orbital_period/500.
''' Note: For batch_1 through batch_5 we used Tf = 4.*orbital_period '''
Tf = 4.*orbital_period
Ntime = int(Tf/dt)
startup = 100

domain = [[-0.7,0.7],[-0.7,0.7]]

def setbox(zbox,vbox):
    plt.xlim(-zbox,zbox)
    plt.ylim(-wbox,wbox)
def initialize_equilibrium_DF(N,vsig,z0):
    u = np.random.uniform(-1.,1.,N)
    x = np.sqrt((1.+u)/(1.-u))
    z = z0*np.log(x)
    z = z - z.mean()
    w = vsig*np.random.normal(0,1,N)
    w = w - w.mean()
    return z,w
def perturb_zw_m2(z,w):
    w = w * 0.5
    return z, w
def perturb_zw_m1(z,w,delta):
    w = w + delta
    return z, w    
def Advance_z(z,w,dt):
    return z + w*dt
def ExternalForce(z,vsig,z0):
    return -2.*vsig*vsig/z0*np.tanh(z/z0)
def SelfGravityForce(N,z,particle_SD):
    indx = np.argsort(z)
    f = np.empty(N)
    I = np.arange(N)
    f[indx] = 2.*np.pi*particle_SD*(N - 2*I - 1 )
    return f
def Advance_w(N,z,w,dt,particle_SD,vsig,z0,live_frac):
    if (live_frac > 0):
        f1 = SelfGravityForce(N,z,particle_SD)
    else:
        f1 = np.zeros(N)
    f2 = ExternalForce(z,vsig,z0)
    f = live_frac*f1 + (1-live_frac)*f2
    return w + f*dt
def RunSimulation(Ntime,N,zeq,weq,dt,particle_SD,vsig,z0,live_frac):
    z = Advance_z(zeq,weq,dt/2)
    w = weq
    zstd = np.zeros(Ntime)
    z_series = np.zeros([Ntime,N])
    w_series = np.zeros([Ntime,N])
    for i in range(Ntime): 
        w = Advance_w(N,z,w,dt,particle_SD,vsig,z0,live_frac)
        z = Advance_z(z,w,dt)
        z_series[i] = z
        w_series[i] = w
        zstd[i] = np.std(z)
    return z,w,zstd, z_series, w_series
def perturb(z,w):
    w *= 1 + np.random.normal(0,0.2)
    z *= 1 + np.random.normal(0,0.2)
    return z, w


animate = 0
check = 0

N_live_frac = 3
files_per_live_frac = 1
snapshots_per_sim = 1

print('Generating '+str(N_live_frac*files_per_live_frac*snapshots_per_sim)+' files.')

index_spacing = (Ntime-startup)//(snapshots_per_sim+1)
index_means = np.arange(1,snapshots_per_sim+1)*index_spacing
index_delta = np.random.normal(0,5,snapshots_per_sim).astype(int)
sample_index = index_means + index_delta


''' Batch Parameters '''
live_frac_values = np.array([0.30123453245,0.50135235,0.701345324]) 
rank = 5
live_frac_index = 0 
start = 0

zeq, weq = initialize_equilibrium_DF(N,vsig,z0)
times = np.arange(0,Ntime)*dt

filename = 0
for live_frac in live_frac_values:
    file_index = 0
    for i in range(0,files_per_live_frac):
        zi, wi = perturb_zw_m1(zeq,weq,0.1)
        z3, w3, z3std, Z, W = RunSimulation(Ntime,N,zi,wi,dt,particle_SD,vsig,z0,live_frac)
        container = dataContainer(data=np.array([Z,W]),delta_t=dt,domain=domain,gridResolution=np.array([60,60]))
        Model = DMD_model(rank,container=container)
        for j in range(0,snapshots_per_sim):
            D = np.hstack([np.hstack([container.density[:,sample_index[j]],np.array([live_frac])]).reshape([container.grid.prod()+1,1]),
                            np.vstack([Model.eigenvectors_DMD,Model.omega.reshape([1,rank])]) ])
            string = 'E:/Research/DMDCNN_suplement/Isothermal/data/rank'+str(rank)+'/dedicated_test_exact_values/run4/'+str(filename+start)
            np.save(string,np.array(D).view(float))
            print(string)
            filename+=1            
        file_index += 1
    live_frac_index += 1

