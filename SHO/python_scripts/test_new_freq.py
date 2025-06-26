import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
from numpy.polynomial.hermite import Hermite

''' Parameters'''
load = False
train = False
test_frac = 0.1 
epochs = 600
learning_rate = 1e-5
validation_split = 0.2
files_per_frequency = 200
max_n = 5

''' Functions '''
def loadData():
    if load:
        print('Loading data into memory.')
        loaded = []
        for omega_index in range(0,3):
            for file_index in range(0,files_per_frequency):
                file_name = '../data/omega'+str(omega_index)+'/sho_'+str(file_index)+'.txt'
                data_file = np.genfromtxt(file_name).view(complex)
                loaded.append(data_file)
        print('Data is in memory.')
        return loaded
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

def H_n(n,x):
    if n>0 :
        order_vector = np.zeros(n,dtype=int)
        order_vector[n-1] = 1        
        hermite_series = Hermite(tuple(order_vector),domain=domain)
    else : hermite_series = Hermite(1,domain=domain)
    return hermite_series(x)

def psi_n(n,omega,x):
    return 1/np.sqrt(2**n*np.math.factorial(n))*(omega/np.pi)**(1/4)*np.exp(-omega*x**2/2)*H_n(n,np.sqrt(omega)*x)

def E_n(n,omega):
    return omega/2*(2*n+1)

def generate_IC(n=max_n,random=False):
    IC = np.zeros(n)
    index = np.arange(1,n)
    if random:
        random_index = np.random.choice(index)
        IC[random_index] = np.random.rand() + 1/2
        IC[0] = 1
        IC = IC/np.sqrt(np.sum(IC**2))
        return IC
    else:
        # return np.ones(n)/np.sqrt(n)        
        IC[0] = 1
        IC[1] = 1
        return IC

def PSI(t):
    exponentials = np.diag(np.exp(-1j*eigenvalues*t))
    return np.dot(np.dot(eigenvectors.T,exponentials),a.T)

''' Load Data '''
if load: 
    loaded = loadData()
    data = np.array(loaded)
    M = data.shape[-1]-1
    r = data.shape[1]-1
    Ntotal = data.shape[0]
    data = np.array(data)
    np.random.shuffle(data)
else: pass

''' Separate into Real and Imaginary components '''
real_data = np.swapaxes(np.array([data.real,data.imag]),0,1)
real_input = real_data[:,:,0,:M]
real_output = real_data[:,:,1:,:M] # ommit eigenvalues 

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
    Gpsi_model.save_weights('../weights/Gpsi_weights.h5')
else: 
    print('Loading eigenvector model weights')
    Gpsi_model.load_weights('../weights/Gpsi_weights.h5')

''' Test Model '''

domain = [-2,2]
x = np.linspace(domain[0],domain[1],M)


''' Generate data with potential curvature outside of those represented in the training data '''
eigenvectors = np.zeros([max_n,M])
eigenvalues = np.zeros(max_n)

new_omega = [3,5]
Nomega = len(new_omega)
new_data = np.zeros([max_n+1,M+1])*0j
new_test_data = np.zeros([len(new_omega),2,M])
for i in range(0,1):
        omega_value = new_omega[i]
        eigenvectors[i] = psi_n(n=i,omega=omega_value,x=x)
        eigenvalues[i] = E_n(n=i,omega=omega_value)
        a = generate_IC(n=max_n,random=True)
        snapshot = PSI(t=5)
        new_data[0,0:M]=snapshot
        new_data[0,M] = omega_value
        new_test_data[i,:,:] = np.array([snapshot.real,snapshot.imag])
        for j in range(0,max_n):
            new_data[1+j,0:M] = eigenvectors[j,:]
            new_data[1+j,M] = eigenvalues[j]
        

''' Test Model '''
model_output = Gpsi_model.predict(test_data).reshape([Ntest,2,r*M]).reshape([Ntest,2,r,M])
new_model_output = Gpsi_model.predict(new_test_data).reshape([Nomega,2,r*M]).reshape([Nomega,2,r,M])

for i in range(0,Nomega):
    for j in range(0,max_n):
        plt.figure()
        plt.plot(x,new_model_output[i,0,j,:],'m')
        plt.plot(x,new_data[1+j,0:M].real,'k')
        plt.savefig('test'+str(new_omega[i])+str(j)+'png')


ape = (np.abs(test_labels+1-(model_output+1)))/np.abs(test_labels+1)*100
print('\nMAPE:{:.2}'.format(ape.mean()),'%')



for i in range(Ntest):
    plt.figure(figsize=(8,8))
    for j in range(0,r):        
        plt.plot(x,model_output[i,0,j,:],'m',label='model output')
        plt.plot(x,test_labels[i,0,j,:],'k',label = 'reference')
    # plt.legend()
    plt.xlabel('x')
    plt.ylabel('$\psi_j(x)$')
    plt.axis([-2,2,-0.6,1.3])
    plt.text(-1.8,1,'$\omega=$'+str(int(test_omega[i])))
    plt.savefig('../figures/model_outputs/eigenvectors/snapshot'+str(i)+'mode.png')
    plt.close()



''' For animation '''
snapshots = test_data[:,0,:]+1j*test_data[:,1,:]
V = model_output[:,0,:,:]+1j*model_output[:,1,:,:]
V_ = test_labels[:,0,:,:]+1j*test_labels[:,1,:,:]

