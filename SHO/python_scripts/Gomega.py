import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
from sys import platform 

''' Parameters'''
load = True
train = True
plot = True
test = False
test_frac = 0.05
epochs = 10000
epochs = 1000
learning_rate = 1e-4
# for 0916 files, patience = 50
patience = 100
validation_split = 0.2
Nomega = 50000
files_per_frequency = 1

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)


''' Functions '''
def loadDataDistribution(file_group,Nomega,files_per_frequency):
    if load:
        print('Loading data into memory.')
        loaded = []
        for omega_index in range(0,Nomega):
            for file_index in range(0,files_per_frequency):
                file_name = '../data/omega_distribution/September16/'+file_group+str(omega_index)+'_'+str(file_index)+'.npy'
                data_file = np.load(file_name).view(complex)
                loaded.append(data_file)
        print('Data is in memory.')
        return np.array(loaded)
    else: print('Data is in memory.')

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
    plt.savefig('../figures/training_curves/MAS'+networkName+'.png',dpi=100)
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
    plt.savefig('../figures/training_curves/MSE'+networkName+'.png',dpi=100)
    plt.close()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

def build_model_val(inputShape,outputShape,activation='relu',learningRate=1e-4,outputActivation='linear'):
    kernel_size = 9
    layer_list = [tf.keras.layers.InputLayer(input_shape=inputShape),  
                  layers.Conv1D(filters=8,kernel_size=kernel_size,activation=activation),
                  layers.Conv1D(filters=8,kernel_size=kernel_size,activation=activation),
                  layers.MaxPool1D(pool_size=2),
                  layers.Conv1D(filters=16,kernel_size=kernel_size,activation=activation),
                  layers.Conv1D(filters=16,kernel_size=kernel_size,activation=activation),
                  layers.MaxPool1D(pool_size=2),
                  layers.Conv1D(filters=32,kernel_size=kernel_size,activation=activation),
                  layers.Conv1D(filters=32,kernel_size=kernel_size,activation=activation),
                  layers.MaxPool1D(pool_size=2),
                  layers.Conv1D(filters=16,kernel_size=kernel_size,activation=activation),
                  layers.Conv1D(filters=16,kernel_size=kernel_size,activation=activation),
                  layers.MaxPool1D(pool_size=2),             
                  layers.Flatten(),
                  layers.Dense(outputShape,activation=outputActivation)]
    model = keras.Sequential(layer_list)
    optimizer = tf.keras.optimizers.Nadam(learningRate)
    model.compile(loss='mae',optimizer=optimizer,metrics=['mae', 'mse'])
    return model

class Percent(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    percent = float(epoch)/epochs*100
    print('\rProgess: {}%'.format(int(percent)), end='')

''' Load Data '''
if load: 
    data = loadDataDistribution(file_group='sho_',Nomega=Nomega,files_per_frequency=files_per_frequency)
else: pass

M = data.shape[-1]-1
r = data.shape[1]-1
domain = [-2,2]
x = np.linspace(domain[0],domain[1],M)
Ntotal = data.shape[0]
data = np.array(data)

Ntest = 0
Ntrain = 50000

''' Separate into Real and Imaginary components '''
real_data = np.swapaxes(np.array([data.real,data.imag]),0,1)
real_input = real_data[:,:,0,:M]
real_output = real_data[:,0,1:,M]  

''' Separate training and test data '''
train_data = np.swapaxes(real_input[:Ntrain,:,:],1,2)
test_data = np.swapaxes(real_input[Ntrain:,:,:],1,2)

input_shape = train_data.shape[1:]
output_shape = np.prod(real_output.shape[1:])


train_labels = real_output[:Ntrain,:]
test_labels = real_output[Ntrain:,:]

train_omega = real_data[:Ntrain,0,0,M]
test_omega = real_data[Ntrain:,0,0,M]

''' Build Model '''
Glambda_model = build_model_val(input_shape,output_shape,learningRate=learning_rate)
Glambda_model.summary()

''' note: Starting with previous weights '''
Glambda_model.load_weights('../weights/Glambda_weights_CNN_0916.h5')


''' Train Model '''
if train:
    print('Training eigenvalue model.')
    history = Glambda_model.fit(train_data,train_labels,epochs=epochs,verbose=0,
                            validation_split=validation_split,callbacks=[Percent(),early_stop])
    history_data = pd.DataFrame(history.history)
    history_data['epoch'] = history.epoch
    history_data.tail()
    plot_history(history,networkName='Glambda')
    Glambda_model.save_weights('../weights/Glambda_weights_CNN_1009.h5')
else:
    print('Loading eigenvalue model weights')
    Glambda_model.load_weights('../weights/Glambda_weights_CNN_0916.h5')
    history_data = pd.read_pickle('Gomega_history.pkl')


if test:

    ''' Test Model '''
    model_output = Glambda_model.predict(test_data).reshape([Ntest,r])
    metrics = Glambda_model.evaluate(test_data,test_labels)
    print('eigenvalue MSE is  '+str(round(metrics[2]/np.abs(test_labels.mean())*100,4))+'%'+' of mean magnitude')

    ''' For animation '''
    E = model_output
    E_ = test_labels

    np.save('ani_E',E)
    np.save('ani_E_',E_)

    if plot:
        for i in range(Ntest):
            plt.figure(figsize=(8,8))
            plt.plot(model_output[i,:],np.zeros(r),'*m',label='model output',markersize=10)
            plt.plot(test_labels[i,:],np.zeros(r),'.k',label = 'reference',markersize=10)
            plt.legend()
            plt.xlabel('Re$\{\lambda\}$')
            plt.ylabel('Im$\{\lambda\}$')
            plt.text(test_labels[i,:].min(),0.04,'$\omega=$'+str(round(test_omega[i],4)))
            plt.savefig('../figures/model_outputs/eigenvalues/snapshot'+str(i)+'mode_new.png')
            plt.close()

