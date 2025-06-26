import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime

''' Parameters'''
load = True
train = False
test_frac = 0.05 
epochs = 600
learning_rate = 1e-5
validation_split = 0.2
Nomega = 100
files_per_frequency = 1

''' Functions '''
def loadDataDistribution(file_group,Nomega,files_per_frequency):
    if load:
        print('Loading data into memory.')
        loaded = []
        for omega_index in range(0,Nomega):
            for file_index in range(0,files_per_frequency):
                file_name = '../data/omega_distribution/June22/'+file_group+str(omega_index)+'_'+str(file_index)+'.txt'
                data_file = np.genfromtxt(file_name).view(complex)
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

def build_model_vec(inputShape,outputShape,activation='relu',learningRate=1e-4,outputActivation='linear'):
    layer_list = [tf.keras.layers.InputLayer(input_shape=inputShape),  
                  layers.Flatten(),
                  layers.Dense(128,activation=activation),
                  layers.Dense(128,activation=activation),
                  layers.Dense(128,activation=activation),
                  layers.Dense(128,activation=activation),
                  layers.Dense(128,activation=activation),
                  layers.Dense(outputShape,activation=outputActivation)]
    model = keras.Sequential(layer_list)
    optimizer = tf.keras.optimizers.Nadam(learningRate)
    model.compile(loss='mse',optimizer=optimizer,metrics=['mae', 'mse'])
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
np.random.shuffle(data)

''' Separate into Real and Imaginary components '''
real_data = np.swapaxes(np.array([data.real,data.imag]),0,1)
real_input = real_data[:,:,0,:M]
real_output = real_data[:,:,1:,:M] 

''' Separate training and test data '''
Ntest = int(test_frac*data.shape[0])
Ntrain = data.shape[0] - Ntest
input_shape = real_input.shape[1:]
output_shape = np.prod(real_output.shape[1:])

train_data = real_input[:Ntrain,:,:]
test_data = real_input[Ntrain:,:,:]

train_labels = real_output[:Ntrain,:,:,:]
test_labels = real_output[Ntrain:,:,:,:]

train_omega = real_data[:Ntrain,0,0,M]
test_omega = real_data[Ntrain:,0,0,M]

'''  Reshape data for use in flat network '''
train_labels = train_labels.reshape([Ntrain,2,r*M])
train_labels = train_labels.reshape([Ntrain,2*r*M])


''' Build Model '''
Gpsi_model = build_model_vec(input_shape,output_shape,learningRate=learning_rate)
Gpsi_model.summary()

''' Train Model '''
if train:
    print('Training eigenvector model.')
    history = Gpsi_model.fit(train_data,train_labels,epochs=epochs,verbose=0,
                            validation_split=validation_split,callbacks=[Percent()])
    history_data = pd.DataFrame(history.history)
    history_data['epoch'] = history.epoch
    history_data.tail()
    plot_history(history,networkName='Gpsi')
    Gpsi_model.save_weights('../weights/Gpsi_weights_June22.h5')
else: 
    print('Loading eigenvector model weights')
    Gpsi_model.load_weights('../weights/Gpsi_weights_June22.h5')

''' Test Model '''
model_output = Gpsi_model.predict(test_data).reshape([Ntest,2,r*M]).reshape([Ntest,2,r,M])

ape = (np.abs(test_labels+1-(model_output+1)))/np.abs(test_labels+1)*100
print('\nMAPE:{:.2}'.format(ape.mean()),'%')

for i in range(Ntest):
    plt.figure(figsize=(8,8))
    for j in range(0,r):        
        plt.plot(x,model_output[i,0,j,:],'m',label='model output')
        plt.plot(x,test_labels[i,0,j,:],'k',label = 'reference')
    plt.xlabel('x')
    plt.ylabel('$\psi_j(x)$')
    plt.axis([-2,2,-0.6,1.3])
    plt.text(-1.8,1,'$\omega=$'+str(round(test_omega[i],4)))
    plt.savefig('../figures/model_outputs/eigenvectors/snapshot'+str(i)+'mode.png')
    plt.close()

''' For animation '''
snapshots = test_data[:1,0,:]+1j*test_data[:,1,:]
V = model_output[:1,0,:,:]+1j*model_output[:,1,:,:]
V_ = test_labels[:1,0,:,:]+1j*test_labels[:,1,:,:]

