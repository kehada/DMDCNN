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

# ''' Functions and Classes '''
class Percent(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    percent = float(epoch)/EPOCHS[model]*100
    print('\rProgess: {}%'.format(int(percent)), end='')
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
def loadData(batches,file_index,N_files,loaded_index):
    print('Loading files...')
    file_type = 'numpy'
    snapshots = []
    batch_index = 0
    for batch in batches:
        print('adding batch '+str(batch))
        internal_file_index = file_index[batch_index]
        # np.random.shuffle(internal_file_index)
        for i in range(N_files-1,0,-1):
            filename = internal_file_index[i]
            loaded_index.append(filename)
            internal_file_index = np.delete(internal_file_index,i)
            if file_type == 'numpy':
                X = np.load('E:/Research/DMDCNN_suplement/Isothermal/data/rank'+str(rank)+'/batch_'+str(batch)+'/'+str(filename)+'.npy').view(complex)
            elif file_type == 'text':
                X = np.genfromtxt('E:/Research/DMDCNN_suplement/Isothermal/data/rank'+str(rank)+'/batch_'+str(batch)+'/'+str(filename)+'.txt').view(complex)
            else :
                print('Invalid file type.')
            if X.shape[1] != rank+1 :
                print('Warning, skipping badly shaped file.')    
                pass
            else:
                snapshots.append(X)      
        file_index[batch_index] = internal_file_index
        batch_index += 1
    data = np.array(snapshots)
    print(str(data.shape[0])+' files loaded.')
    data = data.astype('complex64')
    return data,file_index,loaded_index
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
    model_prediction = model.predict(inputData).reshape([inputData.shape[0],2,int((rank+1)/2)])
    model_prediction[:,1,0] = np.zeros(inputData.shape[0])
    paired = np.zeros([inputData.shape[0],2,rank])
    paired[:,:,:int((rank+1)/2)] = model_prediction
    paired[:,:,int((rank+1)/2):] =  np.tile(np.array([[1,-1],[1,-1]]),inputData.shape[0]).T.reshape([inputData.shape[0],2,2])*model_prediction[:,:,1:]
    paired_eigenvalues = paired[:,0,:] + paired[:,1,:]*1j
    return paired_eigenvalues
def relativeDifference(x,y):
    # return np.abs(x-y)/np.max([x,y])
    return np.abs(x-y)/np.mean([np.abs(x),np.abs(y)])
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

# Flags
train_values = False
train_vectors = False
evaluate = True
plot_results = True
additional_training = False
load_train = False
load_test = True

max_index = 1209

train_files = 1000
test_files = 200

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

# ''' Load Training Data '''
if load_train:
    trained_on = []
    print('\nLoad training data.')
    data,file_index,trained_on = loadData(batches,file_index,train_files,trained_on)
    # np.save('../models/batch_10/trained_on.npy',np.array(trained_on))
    train_input,train_output,train_live_fraction = preprocess(data)
    train_output['vectors'][np.where(np.abs(train_output['vectors'])<1e-4)] = 0

# ''' Train Model '''
if train_vectors:
    print('\nTraining eigenvector model...')  
    model = 'vectors'      
    pre_train = 1
    if pre_train:
        eigenvector_model.load_weights('../models/batch_10/eigenvector_model_0707.h5')

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience['vectors'])

    history = eigenvector_model.fit(train_input, train_output['vectors'],epochs=EPOCHS['vectors'], 
                                validation_split = 0.3, verbose=0,callbacks=[Percent(),early_stop])
    eigenvector_model.save_weights('../models/batch_10/eigenvector_model_1025.h5')   

    histories_vec = []
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    histories_vec.append(hist)
    history_data_vec = pd.concat(histories_vec)
    plotHistory(history_data_vec,networkName='vectors')
    epoch_offset += EPOCHS['vectors']
else :
    # eigenvector_model.load_weights('../models/batch_10/eigenvector_model_0707.h5')
    eigenvector_model.load_weights('../models/batch_10/eigenvector_model_1025.h5')


if train_values:
    print('\n\nTraining eigenvalue model...') 
    model = 'values'       

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience['values'])

    history = eigenvalue_model.fit(train_input, train_output['values'],epochs=EPOCHS['values'], 
                                validation_split = 0.3, verbose=0,callbacks=[Percent(),early_stop])
    eigenvalue_model.save_weights('../models/batch_10/eigenvalue_model_1025.h5')       
    histories_val = []
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    histories_val.append(hist)
    history_data_val = pd.concat(histories_val)
    plotHistory(history_data_val,networkName='values')
    epoch_offset += EPOCHS['values']
else:
    # eigenvalue_model.load_weights('../models/batch_10/eigenvalue_model_0707.h5')
    eigenvalue_model.load_weights('../models/batch_10/eigenvalue_model_1025.h5')


if load_test: 
    # ''' Load Test Data '''
    new = False
    if new:
        tested_on = []
        print('\n\nLoad test data.')
        data,file_index,tested_on = loadData(batches,file_index,test_files,tested_on) 
        # np.save('../models/batch_10/tested_on.npy',np.array(tested_on))
    else : 
        tested_on = np.load('../models/batch_10/tested_on.npy')
        data,tested_on_check = loadTest(tested_on) 

    test_input,test_output,test_live_fraction = preprocess(data)

# ''' Evaluate Model at Test Data '''
if evaluate:
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

# ''' Plot Predictions '''
plot_results = False
if plot_results:
    gridCollection = np.genfromtxt('grid.txt')    
  
        # ax[0,0].set_title(r'$\alpha=$'+str(round(test_live_fraction[i].real,3)))

    for i in range(0,test_input.shape[0]):
        cmap = 'seismic'
        fs = 24
        fig,ax = plt.subplots(rank,4,sharex=True,sharey=True,figsize=(10,9))

        for j in range(0,rank):
            im=ax[j,0].pcolormesh(gridCollection[0],gridCollection[1],test_output['vectors'][i,:].reshape([2,grid.prod(),rank])[0,:,j].reshape(grid),vmin=-0.1,vmax=0.1,cmap=cmap)
            ax[j,2].pcolormesh(gridCollection[0],gridCollection[1],test_output['vectors'][i,:].reshape([2,grid.prod(),rank])[1,:,j].reshape(grid),vmin=-0.1,vmax=0.1,cmap=cmap)            
            ax[j,1].pcolormesh(gridCollection[0],gridCollection[1],test_predictions_vecs[i,0,:,j].reshape(grid),vmin=-0.1,vmax=0.1,cmap=cmap)
            ax[j,3].pcolormesh(gridCollection[0],gridCollection[1],test_predictions_vecs[i,1,:,j].reshape(grid),vmin=-0.1,vmax=0.1,cmap=cmap)
        ax[0,0].set_title(r'True Modes ($Re$)',fontsize=15)
        ax[0,1].set_title(r'Model Output ($Re$)',fontsize=15)
        ax[0,2].set_title(r'True Modes ($Im$)',fontsize=15)
        ax[0,3].set_title(r'Model Output ($Im$)',fontsize=15)
        ax[-1,0].set_xlabel(r'$z$',fontsize=fs)
        ax[-1,0].set_ylabel(r'$v_z$',fontsize=fs)

        for AX in ax:
            for AXS in AX:
                AXS.tick_params(direction='in', length=3, width=1, top=1,bottom=1,left=1,right=1)

        plt.subplots_adjust(hspace=0,wspace=0,right=0.8)

        cbar_ax = fig.add_axes([0.81, 0.25, 0.02, 0.5])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.set_ylabel(r'$Re\{\psi\}$, $Im\{\psi\}$',fontsize=fs)
        ax[0,3].text(-0.5,0.5,r'$\alpha=$'+str(round(test_live_fraction[i].real,3)))
        # plt.savefig('../figures/rank'+str(rank)+'/eigenvectors/batch_10/1025/'+str(i)+'.png',dpi=300)
        plt.savefig('../figures/december/vectors/'+str(i)+'.png',dpi=300)

        plt.close()



    for i in range(0,test_input.shape[0]):
        plt.figure(figsize=(6,6))
        plt.plot(check[i].real,check[i].imag,'ok',label='True Frequencies',markersize=10)
        plt.plot(test_predictions_vals[i].real,test_predictions_vals[i].imag,'*m',label='Model Output',markersize=10)
        plt.xlabel(r'$Re\{\omega\}$',fontsize=22)
        plt.ylabel(r'$Im\{\omega\}$',fontsize=22)
        plt.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
        plt.tick_params(direction='in', length=3, width=1, top=1,bottom=1,left=1,right=1)
        # plt.title(r'$\alpha=$'+str(round(test_live_fraction[i].real,3)))
        plt.tight_layout()
        # plt.savefig('../figures/rank'+str(rank)+'/frequencies/batch_10/1025/'+str(i)+'.png',dpi=300)
        plt.savefig('../figures/december/values/'+str(i)+'.png',dpi=300)

        plt.close()  

gridCollection = np.genfromtxt('grid.txt')    
  
choice = [10,28,31]
choice = [139,46,31,106]
choice = [139,46,31]

plt.figure(figsize=(6,6))

true_colors = ['maroon','darkslategrey','indigo','navy']
model_colors = ['lightcoral','teal','mediumpurple','royalblue']
index = 0

for i in choice :
    plt.plot(check[i].real,check[i].imag,'o',color=true_colors[index],label='True Frequencies',markersize=10)
    plt.plot(test_predictions_vals[i].real,test_predictions_vals[i].imag,'*',color=model_colors[index],label='Model Output',markersize=10)
    plt.xlabel(r'$Re\{\omega\}$',fontsize=22)
    plt.ylabel(r'$Im\{\omega\}$',fontsize=22)
    index += 1
plt.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
plt.tick_params(direction='in', length=3, width=1, top=1,bottom=1,left=1,right=1)
plt.tight_layout()
plt.savefig('../figures/december/multi_december.png',dpi=300)
plt.close()  









label = ['A','B','C']
count = 0

for i in choice:
    cmap = 'seismic'
    fs = 16
    fs_title = 10
    plt.rcParams.update({'font.size': 10})
    fig,ax = plt.subplots(rank,4,sharex=True,sharey=True,figsize=(6,5))

    for j in range(0,rank):
        im=ax[j,0].pcolormesh(gridCollection[0],gridCollection[1],test_output['vectors'][i,:].reshape([2,grid.prod(),rank])[0,:,j].reshape(grid),vmin=-0.1,vmax=0.1,cmap=cmap)
        ax[j,2].pcolormesh(gridCollection[0],gridCollection[1],test_output['vectors'][i,:].reshape([2,grid.prod(),rank])[1,:,j].reshape(grid),vmin=-0.1,vmax=0.1,cmap=cmap)            
        ax[j,1].pcolormesh(gridCollection[0],gridCollection[1],test_predictions_vecs[i,0,:,j].reshape(grid),vmin=-0.1,vmax=0.1,cmap=cmap)
        ax[j,3].pcolormesh(gridCollection[0],gridCollection[1],test_predictions_vecs[i,1,:,j].reshape(grid),vmin=-0.1,vmax=0.1,cmap=cmap)
    
    if i != choice[0]:
        ax[0,0].set_yticklabels([])
        ax[1,0].set_yticklabels([])
        ax[2,0].set_yticklabels([])
        ax[3,0].set_yticklabels([])


    if i == choice[0]:
        ax[0,0].set_title(r'True ($Re$)',fontsize=fs_title)
        ax[0,1].set_title(r'Model ($Re$)',fontsize=fs_title)
        ax[0,2].set_title(r'True ($Im$)',fontsize=fs_title)
        ax[0,3].set_title(r'Model ($Im$)',fontsize=fs_title)
        ax[-1,0].set_xlabel(r'$z$',fontsize=fs)
        ax[-1,0].set_ylabel(r'$v_z$',fontsize=fs)

    for AX in ax:
        for AXS in AX:
            AXS.tick_params(direction='in', length=3, width=0.5, top=1,bottom=1,left=1,right=1)


    # plt.tight_layout()

    plt.subplots_adjust(hspace=0,wspace=0,right=0.8)

    if i == choice[-1]:
        cbar_ax = fig.add_axes([0.81, 0.25, 0.02, 0.5])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.set_ylabel(r'$Re\{\psi\}$, $Im\{\psi\}$',fontsize=fs)


    # ax[0,3].text(-0.5,0.5,r'$\alpha=$'+str(round(test_live_fraction[i].real,3)))
    ax[0,3].text(0.5,0.5,label[count])

    # plt.savefig('../figures/rank'+str(rank)+'/eigenvectors/batch_10/1025/'+str(i)+'.png',dpi=300)
    plt.savefig('../figures/december/vector'+str(i)+'.png',dpi=300)

    plt.close()
    count += 1




