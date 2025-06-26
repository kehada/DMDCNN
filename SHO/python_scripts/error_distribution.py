import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
import scipy.ndimage
from matplotlib import rc
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np

load = 0
load_models = 0
log = False

plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]    
plt.rcParams.update({'font.size': 16})
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def loadDataDistribution(file_group,Nomega,files_per_frequency=1):
    print('Loading data into memory.')
    loaded = []
    for omega_index in range(0,Nomega):
        for file_index in range(0,files_per_frequency):
            file_name = 'E:/Research/DMDCNN_suplement/SHO/data/DedicatedTest2/'+file_group+str(omega_index)+'_'+str(file_index)+'.npy'
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


if load:
    data = loadDataDistribution(file_group='sho_',Nomega=100000)
else : print('data in memory')



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


if load_models:

    Gpsi_model = build_model_vec(input_shape,output_shape)
    Gpsi_model.summary()
    print('Loading eigenvector model weights')
    Gpsi_model.load_weights('../weights/Gpsi_weights_CNN_0916.h5')

    metrics = Gpsi_model.evaluate(test_data,test_labels.reshape([Ntest,r*M]))
    print('eigenvector MSE is  '+str(round(metrics[2]/np.abs(test_labels.mean())*100,4))+'%'+' of mean magnitude')


model_output = Gpsi_model.predict(test_data).reshape([Ntest,r*M]).reshape([Ntest,r,M])
snapshots = test_data[:1,0,:]+1j*test_data[:,1,:]
V = model_output[:,:,:]
V_ = test_labels[:,:,:]


''' VALUES: Separate into Real and Imaginary components '''
real_output = real_data[:,0,1:,M]  

test_data = np.swapaxes(real_input[Ntrain:,:,:],1,2)

input_shape = train_data.shape[1:]
output_shape = np.prod(real_output.shape[1:])

test_labels = real_output[Ntrain:,:]
test_omega = real_data[Ntrain:,0,0,M]

if load_models:

    Glambda_model = build_model_val(input_shape,output_shape)
    print('Loading eigenvalue model weights')
    Glambda_model.load_weights('../weights/Glambda_weights_CNN_1009.h5')

    metrics = Glambda_model.evaluate(test_data,test_labels)
    print('eigenvalue MSE is  '+str(round(metrics[2]/np.abs(test_labels.mean())*100,4))+'%'+' of mean magnitude')

model_output = Glambda_model.predict(test_data).reshape([Ntest,r])
E = model_output
E_ = test_labels

Nbins = 35

resolution_factor = 3

ymin = -3.75
ymax = -1.25
MAE_V = np.log10(np.mean(np.mean(np.abs(V-V_),axis=2),axis=1))

x_edge = np.linspace(1.,9,Nbins)
y_edge = np.linspace(ymin,ymax,Nbins)



H, x_edge, y_edge = np.histogram2d(test_omega,MAE_V,bins=(x_edge,y_edge),normed=False)

x_edges,y_edges = np.meshgrid(x_edge,y_edge)
x_edges = x_edges[:-1,:-1]
y_edges = y_edges[:-1,:-1]

norm = 1/np.sqrt(2*np.pi)*np.exp(-(x_edges)**2/2)


plt.figure(figsize=(8,8))
if log:
    H = np.log10(H+1)
    plt.pcolormesh(x_edges,y_edges,H.T,cmap='Blues')
    plt.contour(scipy.ndimage.zoom(x_edges,resolution_factor),scipy.ndimage.zoom(y_edges,resolution_factor),
    scipy.ndimage.zoom(H.T,resolution_factor),colors='k',levels=np.linspace(0,2.6,10))
else:
    # H = H/norm
    cs=plt.pcolormesh(x_edges,y_edges,H.T,cmap='Blues',vmin=H.min()+5)
    cs.cmap.set_under('w')
    plt.contour(scipy.ndimage.zoom(x_edges,resolution_factor),scipy.ndimage.zoom(y_edges,resolution_factor),
    scipy.ndimage.zoom(H.T,resolution_factor),colors='k',levels=np.linspace(H.min()+5,H.max(),8))


plt.plot(np.ones(2)*4,np.array([ymin,ymax]),'--k')
plt.plot(np.ones(2)*6,np.array([ymin,ymax]),'--k')
plt.plot(np.ones(2)*3,np.array([ymin,ymax]),'--k')
plt.plot(np.ones(2)*2,np.array([ymin,ymax]),'--k')
plt.plot(np.ones(2)*7,np.array([ymin,ymax]),'--k')
plt.plot(np.ones(2)*8,np.array([ymin,ymax]),'--k')
plt.text(4+0.1,ymax-0.1,r'$\sigma_\xi$',fontsize=16)
plt.text(6-0.3,ymax-0.1,r'$\sigma_\xi$',fontsize=16)
plt.text(3.1,ymax-0.1,r'$2\sigma_\xi$',fontsize=16)
plt.text(2.1,ymax-0.1,r'$3\sigma_\xi$',fontsize=16)
plt.text(7-0.4,ymax-0.1,r'$2\sigma_\xi$',fontsize=16)
plt.text(8-0.4,ymax-0.1,r'$3\sigma_\xi$',fontsize=16)
plt.ylim(ymin,ymax)
plt.xlim(2,8)

plt.xlabel(r'$\xi$',fontsize=26)
plt.ylabel(r'$\log_{10}(\ \overline{|\psi-\tilde{\psi}|} \ )$',fontsize=26)
plt.tight_layout()
plt.savefig('../figures/model_outputs/draft_plots/error_distribution_vec.png')
plt.close()


ymin=-2.7
ymax = -0.2

ymin = -2.3
ymax = 0.2



ymin = -2.8
ymax = 0.3




MAE_E = np.log10(np.mean(np.abs(E-E_),axis=1))

x_edge = np.linspace(1.,9,Nbins)
y_edge = np.linspace(ymin,ymax,Nbins)

H, x_edge, y_edge = np.histogram2d(test_omega,MAE_E,bins=(x_edge,y_edge),normed=False)

x_edges,y_edges = np.meshgrid(x_edge,y_edge)
x_edges = x_edges[:-1,:-1]
y_edges = y_edges[:-1,:-1]


plt.figure(figsize=(8,8))

if log:
    H = np.log10(H+1)
    plt.pcolormesh(x_edges,y_edges,H.T,cmap='Blues')
    plt.contour(scipy.ndimage.zoom(x_edges,resolution_factor),scipy.ndimage.zoom(y_edges,resolution_factor),
    scipy.ndimage.zoom(H.T,resolution_factor),colors='k',levels=np.linspace(0,2.6,10))
else :
    # H = H/norm
    cs=plt.pcolormesh(x_edges,y_edges,H.T,cmap='Blues',vmin=H.min()+12)
    cs.cmap.set_under('w')
    plt.contour(scipy.ndimage.zoom(x_edges,resolution_factor),scipy.ndimage.zoom(y_edges,resolution_factor),
    scipy.ndimage.zoom(H.T,resolution_factor),colors='k',levels=np.linspace(H.min()+12,H.max(),6))

plt.plot(np.ones(2)*4,np.array([ymin,ymax]),'--k')
plt.plot(np.ones(2)*6,np.array([ymin,ymax]),'--k')
plt.plot(np.ones(2)*3,np.array([ymin,ymax]),'--k')
plt.plot(np.ones(2)*2,np.array([ymin,ymax]),'--k')
plt.plot(np.ones(2)*7,np.array([ymin,ymax]),'--k')
plt.plot(np.ones(2)*8,np.array([ymin,ymax]),'--k')


label_offset = 0.12

plt.text(4+0.1,ymax-label_offset,r'$\sigma_\xi$',fontsize=16)
plt.text(6-0.3,ymax-label_offset,r'$\sigma_\xi$',fontsize=16)
plt.text(3.1,ymax-label_offset,r'$2\sigma_\xi$',fontsize=16)
plt.text(2.1,ymax-label_offset,r'$3\sigma_\xi$',fontsize=16)
plt.text(7-0.4,ymax-label_offset,r'$2\sigma_\xi$',fontsize=16)
plt.text(8-0.4,ymax-label_offset,r'$3\sigma_\xi$',fontsize=16)
plt.ylim(ymin,ymax)
plt.xlim(2,8)

plt.xlabel(r'$\xi$',fontsize=26)
plt.ylabel(r'$\log_{10}( \ \overline{|\omega-\tilde{\omega}|} \ )$',fontsize=26)
plt.tight_layout()
plt.savefig('../figures/model_outputs/draft_plots/error_distribution_val.png')
plt.close()


