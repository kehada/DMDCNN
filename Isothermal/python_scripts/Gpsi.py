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
train_model = True
checkPredictedModes = True

rank = 5
batches = ['batch_8']
grid = np.array([60,60])

test_frac = 0.1

EPOCHS = 10000
learningRate = 1e-4
inputShape = (60,60,1)

#------------------#
# Pre-Process Data #
#------------------#

def plot_history(history,networkName=''):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(5,5))
  plt.xlabel('Epoch')
  plt.ylabel('MAS')
  plt.plot(hist['epoch'], hist['mae'],'-k',
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],'-m',
           label = 'Val Error')
  plt.legend()
  plt.tight_layout()
  plt.savefig('../figures/rank'+str(rank)+'/training_curves/MAS'+networkName+'.png',dpi=100)
  plt.close()

  plt.figure(figsize=(5,5))
  plt.xlabel('Epoch')
  plt.ylabel('MSE')
  plt.plot(hist['epoch'], hist['mse'],'-k',
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],'-m',
           label = 'Val Error')
  plt.legend()
  plt.axis([hist['epoch'].min(),hist['epoch'].max(),0,hist['val_mse'].max()])
  plt.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
  plt.tight_layout()
  plt.savefig('../figures/rank'+str(rank)+'/training_curves/MSE'+networkName+'.png',dpi=100)
  plt.close()

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


data = cleanZeroFrequency(data)
data = sortZeroFrequency(data)
data = removeUnstable(data)

Ntotal = data.shape[0]
Ntest = int(test_frac*Ntotal)
Ntraining = Ntotal - Ntest

inputShape = np.array([grid[0],grid[1],1])

real_data = np.array([data.real,data.imag])
real_input = real_data[0,:,0:-1,0]
real_output = real_data[:,:,0:-1,1:]
real_output = np.swapaxes(real_output,0,1)
real_output = real_output.reshape([data.shape[0],2*rank*grid.prod()])

normed_train_data = real_input[:Ntraining,:]
normed_train_data = normed_train_data/normed_train_data.max()
normed_train_data = normed_train_data.reshape([Ntraining,60,60,1])
train_labels = real_output[:Ntraining,:]

normed_test_data = real_input[Ntraining:,:]
normed_test_data = normed_test_data/normed_test_data.max()
normed_test_data = normed_test_data.reshape([Ntest,60,60,1])
test_labels = real_output[Ntraining:,:]

live_faction_data = data[:,-1,0]
train_live_fraction = live_faction_data[:Ntraining]
test_live_fraction = live_faction_data[Ntraining:]

#-------------#
# Build Model #
#-------------#

def build_CNN_model(inputShape,outputShape,activation='relu',learningRate=1e-4,outputActivation='linear'):
    
    kernel_size1 = (3,3) 
    layer_list = [tf.keras.layers.InputLayer(input_shape=inputShape),
                  layers.Conv2D(filters=8, kernel_size=kernel_size1),                  
                  layers.Activation(activation='relu'),                  
                  layers.MaxPooling2D((2, 2)),
                  layers.Conv2D(filters=8, kernel_size=kernel_size1),                  
                  layers.Activation(activation='relu'),                  
                  layers.MaxPooling2D((2, 2)),
                  layers.Conv2D(filters=16, kernel_size=kernel_size1),                  
                  layers.Activation(activation='relu'),                  
                  layers.MaxPooling2D((2, 2)),                  
                  layers.Conv2D(filters=16, kernel_size=kernel_size1),                  
                  layers.Activation(activation='relu'),                  
                  layers.MaxPooling2D((2, 2)),   
                  layers.Flatten(),
                  layers.Dense(outputShape,activation=outputActivation)]
    model = keras.Sequential(layer_list)
    optimizer = tf.keras.optimizers.Nadam(learningRate)
    model.compile(loss='mse',optimizer=optimizer,metrics=['mae', 'mse'])
    return model


class Percent(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    percent = float(epoch)/EPOCHS*100
    print('\rProgess: {}%'.format(int(percent)), end='')

eigenvector_model = build_CNN_model(inputShape=inputShape,outputShape=real_output.shape[1],
                                learningRate=learningRate,activation='relu',outputActivation='linear')

eigenvector_model.summary()

#-------------#
# Train Model #
#-------------#

if train_model:
    print('Training eigenvector model')        
    history = eigenvector_model.fit(normed_train_data, train_labels,epochs=EPOCHS, 
                                validation_split = 0.2, verbose=0,callbacks=[Percent()])
    eigenvector_model.save_weights('../models/eigenvector_model_weights_batch_8.h5')

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    plot_history(history,networkName='Vectors')  
else:
    print('Loading model weights')
    eigenvector_model.load_weights('../models/eigenvector_model_weights.h5')

Ncheck = Ntest

test_predictions_vecs = eigenvector_model.predict(normed_test_data[:Ncheck])
test_predictions_vecs = test_predictions_vecs.reshape([Ncheck,2,grid.prod(),rank])    
eigenvectors = test_predictions_vecs[:,0,:,:] + test_predictions_vecs[:,1,:,:]*1j


#---------------#
# Check Outputs #
#---------------#
if checkPredictedModes:
  gridCollection = np.genfromtxt('grid.txt')    
  for i in range(0,int(test_predictions_vecs.shape[0])):
      cmap = 'seismic'
      fs = 24
      fig,ax = plt.subplots(X.shape[1]-1,4,sharex=True,sharey=True,figsize=(12,15))
      for j in range(0,rank):
          ax[j,0].pcolormesh(gridCollection[0],gridCollection[1],test_labels[:Ncheck][i,:].reshape([2,grid.prod(),rank])[0,:,j].reshape(grid),vmin=-0.1,vmax=0.1,cmap=cmap)
          ax[j,1].pcolormesh(gridCollection[0],gridCollection[1],test_labels[:Ncheck][i,:].reshape([2,grid.prod(),rank])[1,:,j].reshape(grid),vmin=-0.1,vmax=0.1,cmap=cmap)            
          ax[j,2].pcolormesh(gridCollection[0],gridCollection[1],test_predictions_vecs[i,0,:,j].reshape(grid),vmin=-0.1,vmax=0.1,cmap=cmap)
          ax[j,3].pcolormesh(gridCollection[0],gridCollection[1],test_predictions_vecs[i,1,:,j].reshape(grid),vmin=-0.1,vmax=0.1,cmap=cmap)
      ax[0,1].set_title('True DMD Modes')
      ax[0,2].set_title('Model Output')
      ax[0,0].set_title(r'$\alpha=$'+str(round(test_live_fraction[i].real,3)))
      ax[-1,0].set_xlabel(r'$z$',fontsize=fs)
      ax[-1,0].set_ylabel(r'$v_z$',fontsize=fs)
      plt.subplots_adjust(hspace=0,wspace=0)
      plt.savefig('../figures/rank'+str(rank)+'/eigenvectors/batch_8/many_snaphot_test'+str(i)+'.png',dpi=300)
      plt.close()


predictions = test_predictions_vecs.reshape(test_labels.shape)
ape = (np.abs(test_labels+1-(predictions+1)))/np.abs(test_labels+1)*100
print('\nMAPE:{:.2}'.format(ape.mean()),'%')