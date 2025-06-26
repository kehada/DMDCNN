import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime

from matplotlib import rc
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]    
plt.rcParams.update({'font.size': 11})
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def loadDataDistribution(file_group,Nomega,files_per_frequency=1):
    print('Loading data into memory.')
    loaded = []
    for omega_index in range(0,Nomega):
        for file_index in range(0,files_per_frequency):
            file_name = '../data/omega_distribution/SpacedTest/'+file_group+str(omega_index)+'_'+str(file_index)+'.npy'
            data_file = np.load(file_name).view(complex)
            loaded.append(data_file)
    print('Data is in memory.')
    return np.array(loaded)


def build_model_vec(inputShape,outputShape,activation='relu',learningRate=1e-4,outputActivation='linear'):
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

data = loadDataDistribution(file_group='sho_',Nomega=9)


M = data.shape[-1]-1
r = data.shape[1]-1
domain = [-2,2]
x = np.linspace(domain[0],domain[1],M)
Ntotal = data.shape[0]
data = np.array(data)

Ntest = Ntotal
Ntrain = 0

''' Separate into Real and Imaginary components '''
real_data = np.swapaxes(np.array([data.real,data.imag]),0,1)
real_input = real_data[:,:,0,:M]
real_output = real_data[:,0,1:,:M] 

input_shape = real_input.shape[1:]
output_shape = np.prod(real_output.shape[1:])


train_data = np.swapaxes(real_input[:Ntrain,:,:],1,2)
test_data = np.swapaxes(real_input[Ntrain:,:,:],1,2)

input_shape = train_data.shape[1:]
output_shape = np.prod(real_output.shape[1:])

train_labels = real_output[:Ntrain,:,:]
test_labels = real_output[Ntrain:,:,:]

train_omega = real_data[:Ntrain,0,0,M]
test_omega = real_data[Ntrain:,0,0,M]

train_labels = train_labels.reshape([Ntrain,r*M])

Gpsi_model = build_model_vec(input_shape,output_shape)
Gpsi_model.summary()
print('Loading eigenvector model weights')
Gpsi_model.load_weights('../weights/Gpsi_weights_CNN_0916.h5')
history_data = pd.read_pickle('Gpsi_history.pkl')

''' Test Model '''
model_output = Gpsi_model.predict(test_data).reshape([Ntest,r*M]).reshape([Ntest,r,M])
metrics = Gpsi_model.evaluate(test_data,test_labels.reshape([Ntest,r*M]))
print('eigenvector MSE is  '+str(round(metrics[2]/np.abs(test_labels.mean())*100,4))+'%'+' of mean magnitude')


''' For animation '''
snapshots = test_data[:1,0,:]+1j*test_data[:,1,:]
V = model_output[:,:,:]
V_ = test_labels[:,:,:]


MAE = np.log10(np.mean(np.mean(np.abs(V-V_),axis=2),axis=1))

x_edge = np.linspace(1.,9,100)
y_edge = np.linspace(-4,-1,100)

H, x_edge, y_edge = np.histogram2d(test_omega,MAE,bins=(x_edge,y_edge))

x_edges,y_edges = np.meshgrid(x_edge,y_edge)



plot_res = False
if plot_res :

    plt.figure()
    plt.pcolormesh(x_edges,y_edges,np.log10(H.T+1),cmap='Greys')
    # plt.plot(np.ones(2)*3,np.array([0,0.04]),'k')
    # plt.plot(np.ones(2)*2,np.array([0,0.04]),'k')
    # plt.plot(np.ones(2)*7,np.array([0,0.04]),'k')
    # plt.plot(np.ones(2)*8,np.array([0,0.04]),'k')

    # plt.text(3.1,0.03,'$2\sigma$')
    # plt.text(2.1,0.03,'$3\sigma$')

    # plt.text(7.1,0.03,'$2\sigma$')
    # plt.text(8.1,0.03,'$3\sigma$')

    # plt.axis([1,9,0,0.04])
    plt.xlabel(r'$V_0$')
    plt.ylabel(r'$\log_{10}(<|\psi-\tilde{\psi}|>)$')
    plt.savefig('error_test.png')
    plt.close()



''' For animation '''
snapshots = test_data[:1,0,:]+1j*test_data[:,1,:]
V = model_output[:,:,:]
V_ = test_labels[:,:,:]

np.save('ani_snapshot',snapshots.view(float))
np.save('ani_V',V)
np.save('ani_V_',V_)

plot_modes = True

if plot_modes:
    for i in range(Ntest):
        plt.figure(figsize=(8,8))
        for j in range(0,r):        
            plt.plot(x,model_output[i,j,:],'m',label='model output')
            plt.plot(x,test_labels[i,j,:],'k',label = 'reference')
        plt.xlabel('q')
        plt.ylabel('$\psi_j(q)$')
        plt.axis([-2,2,-0.6,1.3])
        plt.text(-1.8,1,'$V_0=$'+str(round(test_omega[i],4)))
        plt.savefig('../figures/model_outputs/eigenvectors/snapshot'+str(i)+'mode_new.png')
        plt.close()


