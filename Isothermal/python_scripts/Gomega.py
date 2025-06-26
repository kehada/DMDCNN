import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import datetime
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]    
plt.rcParams.update({'font.size': 16})

''' Parameters'''
train_model = True
checkPredictedModes = True

rank = 5
grid = np.array([60,60])

test_frac = 0.1

EPOCHS  =  2000
learningRate = 1e-4
inputShape = (60,60,1)

#------------------#
# Pre-Process Data #
#------------------#

def removeConjugatePairs(data):
  eigenvalues = data[:,-1,1:]
  no_pair_eigenvalues = np.zeros([data.shape[0],int((rank+1)/2)])*0j
  ind = np.argsort(eigenvalues.imag,axis=1)
  for i in range(0,len(eigenvalues)):
    no_pair_eigenvalues[i,:] = eigenvalues[i,ind[i,:]][int((rank-1)/2):]
  return no_pair_eigenvalues

inputShape = np.array([grid[0],grid[1],1])

real_data = np.array([data.real,data.imag])
real_input = real_data[0,:,0:-1,0]

real_output = np.array([removeConjugatePairs(data).real,removeConjugatePairs(data).imag])
real_output = np.swapaxes(real_output,0,1)
real_output = real_output.reshape([data.shape[0],rank+1])

Ntotal = data.shape[0]
Ntest = int(test_frac*Ntotal)
Ntraining = Ntotal - Ntest

normed_train_data = real_input[:Ntraining,:]
normed_train_data = normed_train_data/normed_train_data.max()
normed_train_data = normed_train_data.reshape([Ntraining,60,60,1])

normed_test_data = real_input[Ntraining:,:]
normed_test_data = normed_test_data/normed_test_data.max()
normed_test_data = normed_test_data.reshape([Ntest,60,60,1])

train_labels = real_output[:Ntraining,:]
test_labels = real_output[Ntraining:,:]

live_faction_data = data[:,-1,0]
train_live_fraction = live_faction_data[:Ntraining]
test_live_fraction = live_faction_data[Ntraining:]


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

class Percent(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    percent = float(epoch)/EPOCHS*100
    print('\rProgess: {}%'.format(int(percent)), end='')

def build_CNN_model(inputShape,outputShape,activation='relu',learningRate=1e-4,outputActivation='linear'):
    
    kernel_size1 = (3,3)
    
    layer_list = [tf.keras.layers.InputLayer(input_shape=inputShape),
                  layers.Conv2D(filters=8, kernel_size=kernel_size1,activation='relu'),                  
                  layers.Conv2D(filters=8, kernel_size=kernel_size1,activation='relu'),
                  layers.MaxPooling2D((2, 2)),                      
                  layers.Conv2D(filters=16, kernel_size=kernel_size1,activation='relu'),
                  layers.Conv2D(filters=16, kernel_size=kernel_size1,activation='relu'),                               
                  layers.MaxPooling2D((2, 2)),                              
                  layers.Flatten(),
                  layers.Dense(outputShape,activation=outputActivation)]
    model = keras.Sequential(layer_list)
    optimizer = tf.keras.optimizers.Nadam(learningRate)
    model.compile(loss='mae',optimizer=optimizer,metrics=['mae', 'mse'])
    return model

#-------------#
# Build Model #
#-------------#
eigenvalue_model = build_CNN_model(activation='relu',inputShape=inputShape,outputShape=int(rank+1),
                                learningRate=learningRate,outputActivation='linear')

eigenvalue_model.summary()

#-------------#
# Train Model #
#-------------#

if train_model:
    print('Training eigenvalue model')
    history = eigenvalue_model.fit(normed_train_data, train_labels,epochs=EPOCHS, validation_split = 0.3,   
                                    verbose=0,callbacks=[Percent()])
    eigenvalue_model.save_weights('../models/frequency_weights.h5')

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    plot_history(history,networkName='Values')  
else:
    print('Loading model weights')
    eigenvalue_model.load_weights('../models/frequency_weights.h5')

loss, mae, mse = eigenvalue_model.evaluate(normed_test_data, test_labels, verbose=2)

Ncheck = Ntest

test_predictions_vals = eigenvalue_model.predict(normed_test_data[0:Ncheck])


def predictEigenvalues(model,inputData):
    model_prediction = model.predict(inputData).reshape([inputData.shape[0],2,int((rank+1)/2)])
    model_prediction[:,1,0] = np.zeros(inputData.shape[0])
    paired = np.zeros([inputData.shape[0],2,rank])
    paired[:,:,:int((rank+1)/2)] = model_prediction
    paired[:,:,int((rank+1)/2):] =  np.tile(np.array([[1,-1],[1,-1]]),Ncheck).T.reshape([Ncheck,2,2])*model_prediction[:,:,1:]
    paired_eigenvalues = paired[:,0,:] + paired[:,1,:]*1j
    return paired_eigenvalues

eigenvalues = predictEigenvalues(eigenvalue_model,normed_test_data[0:Ncheck])
eigenvalue_mags = np.abs(eigenvalues)
for i in range(0,Ncheck):
  eigenvalues[i,:] = eigenvalues[i,:][np.argsort(eigenvalue_mags[i,:])]

#---------------#
# Check Outputs #
#---------------#

check =  data[Ntraining:,-1,1:]
for i in range(0,Ncheck):
    plt.figure(figsize=(6,6))
    plt.plot(check[i].real,check[i].imag,'ok',label='True Frequencies',markersize=10)
    plt.plot(eigenvalues[i].real,eigenvalues[i].imag,'*m',label='Model Output',markersize=10)
    plt.xlabel(r'$Re\{\omega\}$',fontsize=22)
    plt.ylabel(r'$Im\{\omega\}$',fontsize=22)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
    plt.title(r'$\alpha=$'+str(round(test_live_fraction[i].real,3)))
    plt.tight_layout()
    plt.savefig('../figures/rank'+str(rank)+'/frequencies/omega_'+str(i)+'.png',dpi=300)
    plt.close()    

ape = (np.abs(test_labels+1-(test_predictions_vals+1)))/np.abs(test_labels+1)*100

print('MAPE:{:.2}'.format(ape.mean()),'%')


