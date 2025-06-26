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
grid = np.array([60,60])

batches = ['batch_5']
N_live_frac = 5
files_per_live_frac = 1

def cleanZeroFrequency(data):
  eigenvalues = data[:,-1,1:]
  zero_frequency_bool = eigenvalues.imag==0
  zero_frequency_int = zero_frequency_bool.astype(int) 
  zero_frequency_index = np.where(zero_frequency_int.sum(axis=1)==1)[0]
  return data[zero_frequency_index,:,:]
def sortZeroFrequency(data):
    eigenvalues = data[:,-1,1:]
    eigenvalue_mags = np.abs(eigenvalues)
    for i in range(0,data.shape[0]):
        data[i,:,1:] = data[i,:,1:][:,np.argsort(eigenvalue_mags)[i,:]]
    return data
def removeUnstable(data):
    unstable_threshold = 0.9
    live_fraction = data[:,-1,0].real
    unstable_bool = live_fraction>unstable_threshold
    unstable_int = unstable_bool.astype(int) 
    stable_index = np.where(unstable_int==0)[0]
    return data[stable_index,:,:]    


#-----------#
# Load Data #
#-----------#
print('loading snapshots...')
snapshots = []
filenames = []
for batch in batches:
    for i in range(0,N_live_frac):
        for j in range(0,files_per_live_frac):
            filename = 'E:/DMDCNN_suplement/Isothermal/data/rank'+str(rank)+'/'+batch+'/'+'iso_'+str(i)+'_'+str(j)+'.txt'
            X = np.genfromtxt(filename).view(complex)
            if X.shape[1] != rank+1 :
                print('Warning: file: '+filename+' is ill-shapen. Skipping.')    
                pass
            else:
                snapshots.append(X)
                filenames.append(filename)
    print('snapshots in memory')
    test_data = np.array(snapshots)

Ncheck = test_data.shape[0]

test_data = cleanZeroFrequency(test_data)
test_data = sortZeroFrequency(test_data)
test_data = removeUnstable(test_data)

real_test = np.array([test_data.real,test_data.imag])
real_test_input = real_test[0,:,0:-1,0]
real_test_output = real_test[:,:,0:-1,1:]
real_test_output = np.swapaxes(real_test_output,0,1)
real_test_output = real_test_output.reshape([test_data.shape[0],2*rank*grid.prod()])

normed_test_data = real_test_input
normed_test_data = normed_test_data/normed_test_data.max()
normed_test_data = normed_test_data.reshape([test_data.shape[0],grid[0],grid[1],1])
test_labels = real_test_output

live_faction_data = test_data[:,-1,0]
test_live_fraction = live_faction_data

test_predictions_vecs = eigenvector_model.predict(normed_test_data)
test_predictions_vecs = test_predictions_vecs.reshape([Ncheck,2,grid.prod(),rank])    
eigenvectors = test_predictions_vecs[:,0,:,:] + test_predictions_vecs[:,1,:,:]*1j