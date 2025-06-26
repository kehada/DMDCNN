import os
import numpy as np
batch = 6
rank = 5
N_live_frac = 200
files_per_live_frac = 5 

# count = 0
# for i in range(0,N_live_frac):
#     for j in range(0,files_per_live_frac):
#         filename = 'E:/DMDCNN_suplement/Isothermal/data/rank'+str(rank)+'/batch_'+str(batch)+'/'+'iso_'+str(i)+'_'+str(j)+'.txt'
#         os.rename(filename,
#         'E:/DMDCNN_suplement/Isothermal/data/rank'+str(5)+'/batch_'+str(batch)+'/'+str(count)+'.npy')        
#         count += 1


# for count in range(0,1000):
#     os.rename('E:/DMDCNN_suplement/Isothermal/data/rank'+str(5)+'/batch_'+str(batch)+'/'+str(count)+'.npy',
#     'E:/DMDCNN_suplement/Isothermal/data/rank'+str(5)+'/batch_'+str(batch)+'/'+str(count)+'.txt')     



for count in range(0,1000):
    in_file = 'E:/DMDCNN_suplement/Isothermal/data/rank'+str(5)+'/batch_'+str(batch)+'/'+str(count)+'.txt'
    out_file = 'E:/DMDCNN_suplement/Isothermal/data/rank'+str(5)+'/batch_'+str(batch)+'/'+str(count)+'.npy'
    X = np.genfromtxt(in_file).view(complex)
    np.save(out_file,X.view(float))

