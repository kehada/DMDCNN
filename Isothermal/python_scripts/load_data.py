import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import datetime

''' Parameters'''
rank = 5
batches = ['batch_5']
N_live_frac = 5
files_per_live_frac = 1

#-----------#
# Load Data #
#-----------#
print('loading snapshots...')
snapshots = []
filenames = []
for batch in batches:
    for i in range(0,N_live_frac):
        for j in range(0,files_per_live_frac):
            filename = '../data/rank'+str(rank)+'/'+batch+'/'+'iso_'+str(i)+'_'+str(j)+'_'+'.txt'
            X = np.load(filename).view(complex)
            if X.shape[1] != rank+1 :
                print('Warning: file: '+filename+' is ill-shapen. Skipping.')    
                pass
            else:
                snapshots.append(X)
                filenames.append(filename)
    print('snapshots in memory')
    test_data = np.array(snapshots)


