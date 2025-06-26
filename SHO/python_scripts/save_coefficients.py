'''
Run after Gpsi with same namespace
'''
A = np.zeros([Ntest,r])*1j
A_ = np.zeros([Ntest,r])*1j

for j in range(0,Ntest):
    eigenvectors = V[j,:,:]
    eigenvectors_ = V_[j,:,:]

    a = np.dot(np.linalg.pinv(eigenvectors).T,snapshots[j,:])
    a_ = np.dot(np.linalg.pinv(eigenvectors_).T,snapshots[j,:])
    A[j,:] = a
    A_[j,:] = a_

np.save('ani_A',A.view(float))
np.save('ani_A_',A_.view(float))