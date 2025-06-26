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
from scipy import signal
plt.rcParams['animation.ffmpeg_path'] =  'C:\\Users\\Keir\\ffmpeg\\bin\\ffmpeg.exe'
FFwriter = animation.FFMpegWriter()

Model.getSolution()

fig, ax = plt.subplots(2,1)
ims = []
for i in range(0,Ntime,100):
    im1 = ax[0].imshow(container.density[:,i].reshape(container.grid))
    im2 = ax[1].imshow(Model.solution[:,i].real.reshape(container.grid))
    ims.append([im1,im2])
ani = animation.ArtistAnimation(fig, ims, interval=30, blit=False,repeat_delay=1000)      
ani.save('../figures/rank'+str(rank)+'/animations/test.mp4',writer = FFwriter,dpi=300)
plt.close()  