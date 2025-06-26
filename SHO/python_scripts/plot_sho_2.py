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


data = loadDataDistribution(file_group='sho_',Nomega=7)


M = data.shape[-1]-1
r = data.shape[1]-1
domain = [-2,2]
x = np.linspace(domain[0],domain[1],M)
Ntotal = data.shape[0]
data = np.array(data)

Ntest = Ntotal
Ntrain = 0

''' VECTORS: Separate into Real and Imaginary components '''
real_data = np.swapaxes(np.array([data.real,data.imag]),0,1)
real_input = real_data[:,:,0,:M]
real_output = real_data[:,0,1:,:M] 

input_shape = real_input.shape[1:]
output_shape = np.prod(real_output.shape[1:])

train_data = np.swapaxes(real_input[:Ntrain,:,:],1,2)
test_data = np.swapaxes(real_input[Ntrain:,:,:],1,2)

input_shape = train_data.shape[1:]
output_shape = np.prod(real_output.shape[1:])

test_labels = real_output[Ntrain:,:,:]
test_omega = real_data[Ntrain:,0,0,M]

Gpsi_model = build_model_vec(input_shape,output_shape)
Gpsi_model.summary()
print('Loading eigenvector model weights')
Gpsi_model.load_weights('../weights/Gpsi_weights_CNN_0916.h5')

model_output = Gpsi_model.predict(test_data).reshape([Ntest,r*M]).reshape([Ntest,r,M])
metrics = Gpsi_model.evaluate(test_data,test_labels.reshape([Ntest,r*M]))
print('eigenvector MSE is  '+str(round(metrics[2]/np.abs(test_labels.mean())*100,4))+'%'+' of mean magnitude')

snapshots = test_data[:,:,0]+1j*test_data[:,:,1]
V = model_output[:,:,:]
V_ = test_labels[:,:,:]


''' VALUES: Separate into Real and Imaginary components '''
real_output = real_data[:,0,1:,M]  

test_data = np.swapaxes(real_input[Ntrain:,:,:],1,2)

input_shape = train_data.shape[1:]
output_shape = np.prod(real_output.shape[1:])

test_labels = real_output[Ntrain:,:]
test_omega = real_data[Ntrain:,0,0,M]

Glambda_model = build_model_val(input_shape,output_shape)
print('Loading eigenvalue model weights')
Glambda_model.load_weights('../weights/Glambda_weights_CNN_1009.h5')

model_output = Glambda_model.predict(test_data).reshape([Ntest,r])
metrics = Glambda_model.evaluate(test_data,test_labels)
print('eigenvalue MSE is  '+str(round(metrics[2]/np.abs(test_labels.mean())*100,4))+'%'+' of mean magnitude')

E = model_output
E_ = test_labels

choice = [3,5,6]


color_1 = 'tab:grey'
color_2 ='k'

fs = 26

lw1 = 5
lw2 = 2

alpha1=1  
alpha2=1

ls1 = '-'
ls2 = '--'


fig = plt.figure(figsize=(10, 10)) 
gs = gridspec.GridSpec(3, 2, width_ratios=[2, 0.2]) 
ax = []
for j in range(0,3):
    ax_ = []
    for i in range(0,2):
        ax_.append(plt.subplot(gs[j,i]))
    ax.append(ax_)
for j in range(0,r):
    ax[0][0].plot(x,V_[choice[0],j,:],color_1,linewidth=lw1,linestyle=ls1,alpha=alpha1)
    ax[0][0].plot(x,V[choice[0],j,:],color_2,linewidth=lw2,linestyle=ls2,alpha=alpha2)
    ax[0][0].text(-1.5,1,r'$\xi=\bar{\xi} $',fontsize=22)#+str(round(test_omega[choice[0]],3)))

    ax[1][0].plot(x,V_[choice[1],j,:],color_1,linewidth=lw1,linestyle=ls1,alpha=alpha1)
    ax[1][0].plot(x,V[choice[1],j,:],color_2,linewidth=lw2,linestyle=ls2,alpha=alpha2)
    ax[1][0].text(-1.5,1,r'$\xi= \bar{\xi}+2\sigma_\xi$',fontsize=22)#+str(round(test_omega[choice[1]],3)))


    ax[2][0].plot(x,V_[choice[2],j,:],color_1,linewidth=lw1,linestyle=ls1,alpha=alpha1)
    ax[2][0].plot(x,V[choice[2],j,:],color_2,linewidth=lw2,linestyle=ls2,alpha=alpha2)
    ax[2][0].text(-1.5,1,r'$\xi=\bar{\xi}+3\sigma_\xi $',fontsize=22)#+str(round(test_omega[choice[2]],3)))


    ax[0][1].plot(np.linspace(-1,1,3),np.ones(3)*E_[choice[0],j]/test_omega[choice[0]]*2,color_1,linewidth=5)
    ax[0][1].plot(np.linspace(-1,1,3),np.ones(3)*E[choice[0],j]/test_omega[choice[0]]*2,color_2,linestyle='--',linewidth=2)

    ax[1][1].plot(np.linspace(-1,1,3),np.ones(3)*E_[choice[1],j]/test_omega[choice[1]]*2,color_1,linewidth=5)
    ax[1][1].plot(np.linspace(-1,1,3),np.ones(3)*E[choice[1],j]/test_omega[choice[1]]*2,color_2,linestyle='--',linewidth=2)

    ax[2][1].plot(np.linspace(-1,1,3),np.ones(3)*E_[choice[2],j]/test_omega[choice[2]]*2,color_1,linewidth=5)
    ax[2][1].plot(np.linspace(-1,1,3),np.ones(3)*E[choice[2],j]/test_omega[choice[2]]*2,color_2,linestyle='--',linewidth=2)        

ax[0][1].set_ylabel(r'$2i\omega_j/\xi$',fontsize=fs)
ax[1][1].set_ylabel(r'$2i\omega_j/\xi$',fontsize=fs)
ax[2][1].set_ylabel(r'$2i\omega_j/\xi$',fontsize=fs)
ax[2][1].set_xlabel('',fontsize=fs)

ax[0][0].tick_params(direction='in', length=3, width=1, top=1,bottom=1,left=1,right=1)
ax[1][0].tick_params(direction='in', length=3, width=1, top=1,bottom=1,left=1,right=1)
ax[2][0].tick_params(direction='in', length=3, width=1, top=1,bottom=1,left=1,right=1)

ax[0][1].tick_params(direction='in', length=3, width=1, top=1,bottom=1,left=1,right=1)
ax[1][1].tick_params(direction='in', length=3, width=1, top=1,bottom=1,left=1,right=1)
ax[2][1].tick_params(direction='in', length=3, width=1, top=1,bottom=1,left=1,right=1)

ax[2][0].set_xlabel(r'$q$',fontsize=fs)
ax[0][0].set_ylabel(r'$\psi_j(q)$',fontsize=fs)
ax[1][0].set_ylabel(r'$\psi_j(q)$',fontsize=fs)
ax[2][0].set_ylabel(r'$\psi_j(q)$',fontsize=fs)
for i in range(0,3):
    ax[i][0].set_xlim([-1.7,1.7])
    ax[i][0].set_ylim([-0.5,1.3])
    if i <2:
        ax[i][0].xaxis.set_ticklabels([])
    ax[i][1].xaxis.set_visible(False)
# plt.tight_layout()
plt.subplots_adjust(hspace=0.1,wspace=0.3)
plt.savefig('../figures/model_outputs/draft_plots/test.png',dpi=300)
plt.close()


