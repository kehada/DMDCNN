import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import datetime


Nfiles = 299
rank = 5
batches = ['batch_0']
N_live_frac = 10
files_per_live_frac = 2


grid = np.array([60,60])

load_data = True
train_model = False
checkPredictedModes = False

#-----------#
# Load Data #
#-----------#
if load_data:
    print('loading snapshots...')
    snapshots = []
    filenames = []
    for batch in batches:
        for i in range(0,N_live_frac):
          for j in range(0,files_per_live_frac):
            filename = '../data/rank'+str(rank)+'/'+batch+'/'+'iso_'+str(i)+'_'+str(j)+'.txt'
            X = np.genfromtxt(filename).view(complex)
            if X.shape[1] != rank+1 :
                print('Warning: file: '+filename+' is ill-shapen. Skipping.')    
                pass
            else:
                snapshots.append(X)
                filenames.append(filename)
    print('snapshots in memory')
    data = np.array(snapshots)
    np.random.seed(1)
    np.random.shuffle(data)

else: pass
