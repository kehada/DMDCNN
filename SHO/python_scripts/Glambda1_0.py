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
train = False
test_frac = 0.1 
epochs = 400
learning_rate = 1e-4
validation_split = 0.2
files_per_frequency = 200

''' Functions '''
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

def build_model_val(inputShape,outputShape,activation='relu',learningRate=1e-4,outputActivation='linear'):
    layer_list = [tf.keras.layers.InputLayer(input_shape=inputShape),  
                  layers.Flatten(),
                  layers.Dense(512,activation=activation),
                  layers.Dense(512,activation=activation),
                  layers.Dense(256,activation=activation),
                  layers.Dense(256,activation=activation),
                  layers.Dense(128,activation=activation),
                  layers.Dense(128,activation=activation),
                  layers.Dense(64,activation=activation),
                  layers.Dense(64,activation=activation),
                  layers.Dense(32,activation=activation),
                  layers.Dense(32,activation=activation),
                  layers.Dense(outputShape,activation=outputActivation)]
    model = keras.Sequential(layer_list)
    optimizer = tf.keras.optimizers.Nadam(learningRate)
    model.compile(loss='mae',optimizer=optimizer,metrics=['mae', 'mse'])
    return model

class Percent(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    percent = float(epoch)/epochs*100
    print('\rProgess: {}%'.format(int(percent)), end='')

''' Separate into Real and Imaginary components '''
real_data = np.swapaxes(np.array([data.real,data.imag]),0,1)
real_input = real_data[:,:,0,:M]
real_output = real_data[:,0,1:,M]  

''' Separate training and test data '''
Ntest = int(test_frac*data.shape[0])
Ntrain = data.shape[0] - Ntest
input_shape = real_input.shape[1:]
output_shape = np.prod(real_output.shape[1:])

train_data = real_input[:Ntrain,:,:]
test_data = real_input[Ntrain:,:,:]

train_labels = real_output[:Ntrain,:]
test_labels = real_output[Ntrain:,:]

train_omega = real_data[:Ntrain,0,0,M]
test_omega = real_data[Ntrain:,0,0,M]

''' Build Model '''
Glambda_model = build_model_val(input_shape,output_shape,learningRate=learning_rate)
Glambda_model.summary()

''' Train Model '''
if train:
    print('Training eigenvalue model.')
    history = Glambda_model.fit(train_data,train_labels,epochs=epochs,verbose=0,
                            validation_split=validation_split,callbacks=[Percent()])
    history_data = pd.DataFrame(history.history)
    history_data['epoch'] = history.epoch
    history_data.tail()
    plot_history(history,networkName='Glambda')
    Glambda_model.save_weights('../weights/Glambda_weights.h5')
else:
    print('Loading eigenvalue model weights')
    Glambda_model.load_weights('../weights/Glambda_weights.h5')

''' Test Model '''
model_output = Glambda_model.predict(test_data).reshape([Ntest,r])

ape = (np.abs(test_labels+1-(model_output+1)))/np.abs(test_labels+1)*100
print('\nMAPE:{:.2}'.format(ape.mean()),'%')


for i in range(2):
    plt.figure(figsize=(8,8))
    plt.plot(model_output[i,:],np.zeros(r),'*m',label='model output',markersize=10)
    plt.plot(test_labels[i,:],np.zeros(r),'.k',label = 'reference',markersize=10)
    plt.legend()
    plt.xlabel('Re$\{\lambda\}$')
    plt.ylabel('Im$\{\lambda\}$')
    plt.text(test_labels[i,:].min(),0.04,'$\omega=$'+str(int(test_omega[i])))
    plt.savefig('../figures/model_outputs/eigenvalues/snapshot'+str(i)+'.png')
    plt.close()


''' For animation '''
E = model_output
E_ = test_labels

