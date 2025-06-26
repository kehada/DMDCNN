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
import sparse
from scipy.stats import binned_statistic


# ''' Functions and Classes '''
def plotHistory(hist,networkName=''):
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
def removeConjugatePairs(data):
  eigenvalues = data[:,-1,1:]
  no_pair_eigenvalues = np.zeros([data.shape[0],int((rank+1)/2)])*0j
  ind = np.argsort(eigenvalues.imag,axis=1)
  for i in range(0,len(eigenvalues)):
    no_pair_eigenvalues[i,:] = eigenvalues[i,ind[i,:]][int((rank-1)/2):]
  return no_pair_eigenvalues
def build_eigenvector_model(inputShape,outputShape,activation='relu',learningRate=1e-4,outputActivation='linear'):
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
    model.compile(loss='mae',optimizer=optimizer,metrics=['mae', 'mse'])
    return model
def build_eigenvalue_model(inputShape,outputShape,activation='relu',learningRate=1e-4,outputActivation='linear'):
    
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
def loadDataTest(N_files):
    print('Loading files...')
    file_type = 'numpy'
    snapshots = []

    for i in range(0,N_files,):
        filename = i
        if file_type == 'numpy':
            X = np.load('E:/Research/DMDCNN_suplement/Isothermal/data/rank'+str(rank)+'/dedicated_test_exact_values/run4/'+str(filename)+'.npy').view(complex)
        if X.shape[1] != rank+1 :
            print('Warning, skipping badly shaped file.')    
            pass
        else:
            snapshots.append(X)      

    data = np.array(snapshots)
    print(str(data.shape[0])+' files loaded.')
    data = data.astype('complex64')
    return data
def preprocess(data):
    data = cleanZeroFrequency(data)
    data = sortZeroFrequency(data)
    data = removeUnstable(data)

    real_data = np.array([data.real,data.imag])
    real_input = real_data[0,:,0:-1,0]
    real_output_vec = real_data[:,:,0:-1,1:]
    real_output_vec = np.swapaxes(real_output_vec,0,1)
    real_output_vec = real_output_vec.reshape([data.shape[0],2*rank*grid.prod()])

    real_output_val = np.array([removeConjugatePairs(data).real,removeConjugatePairs(data).imag])
    real_output_val = np.swapaxes(real_output_val,0,1)
    real_output_val = real_output_val.reshape([data.shape[0],rank+1])

    normed_data = real_input
    normed_data = normed_data/normed_data.max()
    normed_data = normed_data.reshape([data.shape[0],grid[0],grid[1],1])
    live_faction_data = data[:,-1,0]

    output = {'vectors':real_output_vec,'values':real_output_val}

    return normed_data,output,live_faction_data
def predictEigenvalues(model,inputData):
    print(model.predict(inputData).shape)
    model_prediction = model.predict(inputData).reshape([inputData.shape[0],2,int((rank+1)/2)])
    print(model_prediction.shape)
    model_prediction[:,1,0] = np.zeros(inputData.shape[0])
    paired = np.zeros([inputData.shape[0],2,rank])
    paired[:,:,:int((rank+1)/2)] = model_prediction
    paired[:,:,int((rank+1)/2):] =  np.tile(np.array([[1,-1],[1,-1]]),inputData.shape[0]).T.reshape([inputData.shape[0],2,2])*model_prediction[:,:,1:]
    paired_eigenvalues = paired[:,0,:] + paired[:,1,:]*1j
    return paired_eigenvalues
def loadTest(file_index):
    loaded = []
    snapshots = []
    for i in file_index:
        X = np.load('E:/Research/DMDCNN_suplement/Isothermal/data/rank'+str(5)+'/batch_'+str(10)+'/'+str(i)+'.npy').view(complex)
        loaded.append(i)
        snapshots.append(X)
    data = np.array(snapshots)
    data = data.astype('complex64')
    return data,loaded


load_test = True

max_index = 2

batches = [10]

file_index = []
for i in range(0,len(batches)):
    file_index.append(np.arange(0,max_index))

rank = 5
grid = np.array([60,60])

EPOCHS = {'vectors':5000,
          'values' :2000}

patience = {'vectors':200,
          'values' :50}

learningRate = {'vectors':5e-5,
                'values' :1e-4}

epoch_offset = 0 
inputShape = (60,60,1)
outputShape = [2*rank*grid.prod(),int(rank+1)]

# ''' Build Model '''
eigenvector_model = build_eigenvector_model(inputShape=inputShape,outputShape=outputShape[0],
                                    learningRate=learningRate['vectors'],activation='relu',outputActivation='linear')
eigenvector_model.summary()

eigenvalue_model = build_eigenvalue_model(inputShape=inputShape,outputShape=outputShape[1],
                                    learningRate=learningRate['values'],activation='relu',outputActivation='linear')
eigenvalue_model.summary()

# ''' Load Models '''
eigenvector_model.load_weights('../models/batch_10/eigenvector_model_1025.h5')
eigenvalue_model.load_weights('../models/batch_10/eigenvalue_model_1025.h5')

if load_test: 
    # ''' Load Test Data '''
    print('\n\nLoad test data.')
    data = loadDataTest(max_index) 
    test_input,test_output,test_live_fraction = preprocess(data)

# ''' Evaluate Model at Test Data '''
test_predictions_vecs = eigenvector_model.predict(test_input)
test_predictions_vecs = test_predictions_vecs.reshape([test_input.shape[0],2,grid.prod(),rank])    
eigenvectors = test_predictions_vecs[:,0,:,:] + test_predictions_vecs[:,1,:,:]*1j
predictions = test_predictions_vecs.reshape(test_output['vectors'].shape)
print('\n')
metrics = eigenvector_model.evaluate(test_input,test_output['vectors'])
print('\n')
print('eigenvector MSE is  '+str(round(metrics[2]/np.abs(test_output['vectors']).mean()*100,4))+'%'+' of mean magnitude')
print('\n')

check =  data[:,-1,1:]
test_predictions_vals = predictEigenvalues(eigenvalue_model,test_input)
eigenvalue_mags = np.abs(test_predictions_vals)
for i in range(0,test_input.shape[0]):
    test_predictions_vals[i,:] = test_predictions_vals[i,:][np.argsort(eigenvalue_mags[i,:])]

metrics = eigenvalue_model.evaluate(test_input,test_output['values'])
print('\n')
print('eigenvalue MSE is  '+str(round(metrics[2]/np.abs(test_output['values']).mean()*100,4))+'%'+' of mean magnitude')
print('\n')


test_labels = test_output['vectors']


gridCollection = np.genfromtxt('grid.txt')    
for i in range(0,int(test_predictions_vecs.shape[0])):
    cmap = 'seismic'
    fs = 24
    fig,ax = plt.subplots(5,4,sharex=True,sharey=True,figsize=(10,9))
    for j in range(0,rank):
        im=ax[j,0].pcolormesh(gridCollection[0],gridCollection[1],test_output['vectors'][i,:].reshape([2,grid.prod(),rank])[0,:,j].reshape(grid),vmin=-0.1,vmax=0.1,cmap=cmap)
        ax[j,2].pcolormesh(gridCollection[0],gridCollection[1],test_output['vectors'][i,:].reshape([2,grid.prod(),rank])[1,:,j].reshape(grid),vmin=-0.1,vmax=0.1,cmap=cmap)            
        ax[j,1].pcolormesh(gridCollection[0],gridCollection[1],test_predictions_vecs[i,0,:,j].reshape(grid),vmin=-0.1,vmax=0.1,cmap=cmap)
        ax[j,3].pcolormesh(gridCollection[0],gridCollection[1],test_predictions_vecs[i,1,:,j].reshape(grid),vmin=-0.1,vmax=0.1,cmap=cmap)
    ax[0,0].set_title(r'True Modes ($Re$)',fontsize=15)
    ax[0,1].set_title(r'Model Output ($Re$)',fontsize=15)
    ax[0,2].set_title(r'True Modes ($Im$)',fontsize=15)
    ax[0,3].set_title(r'Model Output ($Im$)',fontsize=15)
    ax[0,3].text(-0.5,0.5,r'$\alpha=$'+str(round(test_live_fraction[i].real,3)))
    ax[-1,0].set_xlabel(r'$z$',fontsize=fs)
    ax[-1,0].set_ylabel(r'$v_z$',fontsize=fs)

    for AX in ax:
        for AXS in AX:
            AXS.tick_params(direction='in', length=3, width=1, top=1,bottom=1,left=1,right=1)
    plt.subplots_adjust(hspace=0,wspace=0,right=0.8)

    cbar_ax = fig.add_axes([0.81, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel(r'$Re\{\psi\}$, $Im\{\psi\}$',fontsize=fs)
    plt.savefig('../figures/dedicated_test/eigenvector'+str(i)+'.png',dpi=300)
    plt.close()



