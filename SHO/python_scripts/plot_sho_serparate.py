from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


'''
eigenvectors
'''
fig,ax = plt.subplots(3,1,figsize=(4,9),sharey=True,sharex=True)
for j in range(0,r):
    ax[0].plot(x,V_[17,j,:],'k',linewidth=2)
    ax[0].plot(x,V[17,j,:],':m',linewidth=2)
    ax[0].text(-1.5,1,'$\omega=$'+str(round(test_omega[17],4)))

    ax[1].plot(x,V_[120,j,:],'k',linewidth=2)
    ax[1].plot(x,V[120,j,:],':m',linewidth=2)
    ax[1].text(-1.5,1,'$\omega=$'+str(round(test_omega[120],4)))


    ax[2].plot(x,V_[119,j,:],'k',linewidth=2)
    ax[2].plot(x,V[119,j,:],':m',linewidth=2)
    ax[2].text(-1.5,1,'$\omega=$'+str(round(test_omega[119],4)))


ax[2].set_xlabel('$q$',fontsize=20)
ax[0].set_ylabel('$\\psi_j$',fontsize=20)
ax[1].set_ylabel('$\\psi_j$',fontsize=20)
ax[2].set_ylabel('$\\psi_j$',fontsize=20)
plt.axis([-1.7,1.7,-0.5,1.3])
plt.tight_layout()
plt.subplots_adjust(hspace=0.1)
plt.savefig('../figures/model_outputs/draft_plots/eigenvectors_test.png',dpi=100)
plt.close()


'''
eigenvalues
'''

fig,ax = plt.subplots(3,1,figsize=(1.2,9),sharey=True,sharex=True)
for j in range(0,5):
    ax[0].plot(np.linspace(-1,1,3),np.ones(3)*E_[17,j]/test_omega[17]*2,'k',linewidth=2)
    ax[0].plot(np.linspace(-1,1,3),np.ones(3)*E[17,j]/test_omega[17]*2,':m',linewidth=2)

    ax[1].plot(np.linspace(-1,1,3),np.ones(3)*E_[120,j]/test_omega[120]*2,'k',linewidth=2)
    ax[1].plot(np.linspace(-1,1,3),np.ones(3)*E[120,j]/test_omega[120]*2,':m',linewidth=2)

    ax[2].plot(np.linspace(-1,1,3),np.ones(3)*E_[119,j]/test_omega[119]*2,'k',linewidth=2)
    ax[2].plot(np.linspace(-1,1,3),np.ones(3)*E[119,j]/test_omega[119]*2,':m',linewidth=2)        
    plt.xlim([-1,1])
    ax[0].set_ylabel('$2i\omega_j/V_0$',fontsize=20)
    ax[1].set_ylabel('$2i\omega_j/V_0$',fontsize=20)
    ax[2].set_ylabel('$2i\omega_j/V_0$',fontsize=20)
    ax[0].xaxis.set_visible(False)
    ax[1].xaxis.set_visible(False)
    ax[2].xaxis.set_visible(False)
    ax[2].set_xlabel('',fontsize=20)


    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
plt.savefig('../figures/model_outputs/draft_plots/eigenvalues_test.png',dpi=100)    
plt.close()