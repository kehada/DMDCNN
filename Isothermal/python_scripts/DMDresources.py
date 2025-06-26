import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]    
plt.rcParams.update({'font.size': 16})
import pickle

#---------#
# Classes #
#---------#
class dataContainer:
    def __init__(self,data,delta_t,domain,coordinateLabels=None,observableType='density',gridResolution=None):
        '''
        data: 
            numpy array of shape (dimension, Ntimes, Nmeasurements)
            - dimension is the dimension of the data space for each measurement. For example if the data
            comes from a simulation of 10^6 stars, and each star has 6 phase space coordinates tracked 
            in the simulation, dimension is 6.
            - Ntimes is the number of time samples
            - Nmeasurements can be the number of stars, number of sensors, etc. 
        delta_t:
            sampling period of the descrete time snapshots in the data.
        coordinateLabels:
            string of labels for each of the dimensions of the data.
            - for example, for a 4D phase space, one might use: coordinateLabels = ['q_1','q_2','p_1','p_2']
        observableType: 
            string describing the type of observable to be used for DMD. As of now only density is implemented here.
            An observable type denoted function will be added in the future, which will allow the user to pass this class 
            a function that maps the raw data to a desired observable space.
        gridResolution:
            array of the number of grid points per dimension
            - if no argument is provided, the default resolution of 30 bins per dimension will be used. 
        '''

        self.delta_t = delta_t
        self.observableType = observableType
        self.dimension = data.shape[0]
        self.Ntimes = data.shape[1]
        self.Nmeasurements = data.shape[2]
        self.domain = domain
        
        if type(coordinateLabels) == type(None):
            # generate generic labels
            self.coordinateLabels = []
            for i in range(0,self.dimension):
                self.coordinateLabels.append('x_'+str(i+1))
        if type(gridResolution) == type(None):
            self.grid = np.ones(self.dimension)*60
            self.grid = self.grid.astype(int)
        else: 
            try :
                if len(gridResolution)==self.dimension:
                    self.grid = gridResolution
            except: 
                print('Warning: invalid gridResolution argument provided. Using default values.')
                self.grid = np.ones(self.dimension)*30
                self.grid = self.grid.astype(int)

        if observableType == 'density' :
            self.norms = np.max(np.abs(data.reshape([self.dimension,self.Ntimes*self.Nmeasurements])),axis=1)
            normMatrix = np.tile(self.norms,[int(data.shape[1]*data.shape[2]),1]).T            
            data = data.reshape([self.dimension,int(data.shape[1]*data.shape[2])])/normMatrix
            self.data = data.reshape([self.dimension,self.Ntimes,self.Nmeasurements])
            self.density = np.zeros(np.hstack([self.Ntimes,self.grid]))
            binEdges = []
            self.gridCollection = []
            for i in range(0,self.dimension):
                binEdges.append(np.linspace(self.domain[i][0],self.domain[i][1],self.grid[i]+1))
            for elements in binEdges :
                self.gridCollection.append((elements[1:]+elements[0:-1])/2)
            for i in range(0,self.Ntimes):
                self.density[i],self.edges = np.histogramdd(self.data[:,i,:].T,binEdges)
            self.density = self.density.T.reshape([self.grid.prod(),self.Ntimes])
                
class DMD_model:
    '''
    In the current implementation, the best way to use this class is by initilizing it by simply passing a dataContainer object, 
    and rank. You can provide each of the components the dataContainer holds individually if you do not want to use dataContainer.
    However, if your desired DMD modes are of density, the dataContainer class will properly prepare your time series data for the 
    methods in the DMD_model class. 
    '''
    def __init__(self,rank=None,data=None,grid=None,gridCollection=None,delta_t=None,container=None):
        
        if rank == None:
            print('Warning: No rank provided. Using default value of r=15')
            self.rank = 15
        else:
            if type(rank) != int:
                print('Warning: Invalid rank argument provided, it must be an integer. Using default value of r=15')
                self.rank = 15
            else:
                self.rank = rank
        
        if container == None :
            if type(data) == type(None): 
                raise TypeError('Error: If a container is not provided, the arguments: data, grid, gridCollection, and delta_t must be.')                
            if delta_t == None: 
                raise TypeError('Error: If a container is not provided, the arguments: data, grid, gridCollection, and delta_t must be.')                
            if type(grid) == type(None): 
                raise TypeError('Error: If a container is not provided, the arguments: data, grid, gridCollection, and delta_t must be.')                                
            if type(gridCollection) == type(None): 
                raise TypeError('Error: If a container is not provided, the arguments: data, grid, gridCollection, and delta_t must be.')                                
            
            
            self.data = data
            self.delta_t = delta_t
            self.grid = grid
            self.gridCollection = gridCollection
        if container != None :
            try:
                self.data = container.density
                self.grid = container.grid
                self.gridCollection = container.gridCollection
                self.delta_t = container.delta_t
            except:
                raise TypeError('Error: Given container is invalid. Make sure you generate the container from the dataContainer class.')                                
    
        self.X = self.data[:,0:-1]
        self.Xprime = self.data[:,1:]
        self.dimension = len(self.grid)
        U, self.singular_values, V_conjugate = np.linalg.svd(self.X, full_matrices=False) 
        U = U[:,0:self.rank]
        self.singular_values = self.singular_values[0:self.rank]
        V_conjugate = V_conjugate[0:self.rank,:]
        self.V = np.matrix(V_conjugate).getH()
        self.U = np.matrix(U)
        self.SV_inverse = np.diag(1/self.singular_values)
        self.A_tilde = np.dot(self.U.getH(),np.dot(self.Xprime,np.dot(self.V,self.SV_inverse)))
        eigenvalues_A_tilde, eigenvectors_A_tilde = np.linalg.eig(self.A_tilde)
        self.eigenvectors_DMD  = np.dot(self.Xprime,np.dot(self.V,
                                 np.dot(self.SV_inverse,eigenvectors_A_tilde)))
        self.eigenvalues_DMD = eigenvalues_A_tilde
        self.omega = np.log(self.eigenvalues_DMD)/self.delta_t        
        self.t = np.arange(0,self.data.shape[1])*self.delta_t

    def getAmplitudes(self,t=None):
        if t==None:
            t=self.t
        self.a = np.zeros([self.rank,len(t)])*0j
        b = np.squeeze(np.array(np.dot(np.array(np.linalg.pinv(self.eigenvectors_DMD)),self.X[:,0]).T))
        for i in range(0,self.rank):
            self.a[i] = np.exp(self.omega[i]*t)*b[i]
        self.amplitudeEnergy = np.zeros(self.rank)
        for i in range(0,self.rank):
            self.amplitudeEnergy[i] = np.sum(np.abs(self.a[i])**2)

    def getSolution(self,t=None):
        if t == None :
            t = self.t
        self.solution = np.zeros([len(self.X[:,0]),len(t)])*0j
        for i in range(0,len(t)):
            b = np.dot(np.linalg.pinv(self.eigenvectors_DMD),self.X[:,0]).T
            self.solution[:,i] = np.squeeze(np.array( np.dot(np.dot(self.eigenvectors_DMD,
            np.diag(np.exp(self.omega*t[i]))),b ) ))    
        

    def plotSolution(self,log=False):
        self.getSolution()
        if self.dimension == 2:
            fig = plt.figure()
            ims = []
            for i in range(0,self.data.shape[1]):
                plotData = self.solution[:,i].reshape(self.grid).real
                im = plt.pcolormesh(self.gridCollection[0],self.gridCollection[1],plotData,cmap='Greys',vmin = 0, vmax = self.solution.max().real)
                ims.append(im)

        if self.dimension > 2:
            fig = plt.figure()
            axes = generatePlotAxes(self.dimension)
            indices = generatePlotIndices(self.dimension)
            ims = []
            for i in range(0,self.data.shape[1]):                
                subIms = []
                for j in range(0,len(axes)):
                    sumIndices = list(np.arange(0,self.dimension))
                    for elements in indices[j]:
                        sumIndices.remove(elements)
                    sumIndices = tuple(sumIndices)                    
                    plotData = np.sum(self.solution[:,i].reshape(self.grid).real,axis=sumIndices)
                    im = axes[j].pcolormesh(self.gridCollection[indices[j][0]],self.gridCollection[indices[j][1]],plotData,cmap='Greys',vmin = 0, vmax = self.solution.max().real)
                    subIms.append(im)
                plt.subplots_adjust(wspace=1/2,hspace=1/2)
                ims.append(subIms)

        ani = animation.ArtistAnimation(fig, ims, interval=90, blit=False,repeat_delay=1000)
        plt.tight_layout()
        ani.save('../figures/solution.gif',dpi=300)
        plt.close()

    def plotModes(self,labels=None,modeRange=None):
        if modeRange == None :
            modeMin = -1
            modeMax = 1
        else:
            if len(modeRange) == 2:
                modeMin,modeMax = modeRange
            else:
                print('Warning: argument modeRange must be a two element list of the form [min,max]. Using default range.')
                modeMin = -1
                modeMax = 1                
                
        self.basicDiagnostics()
        for i in range(0,self.rank):
            if self.dimension < 2:
                plt.figure(figsize=(10,10))
                plt.plot(self.gridCollection,self.allInfo[self.sliceMap['eigenvectors']][i])
            else : 
                # Real components of modes
                plt.figure(figsize=(10,10))
                axes = generatePlotAxes(self.dimension)
                indices = generatePlotIndices(self.dimension)
                for j in range(0,len(axes)):
                    ax = axes[j]
                    sumIndices = list(np.arange(0,self.dimension))
                    for elements in indices[j]:
                        sumIndices.remove(elements)
                    sumIndices = tuple(sumIndices)
                    plotData = np.sum(self.allInfo[self.sliceMap['eigenvectors']][i].reshape(self.grid).real,axis=sumIndices)
                    ax.pcolormesh(self.gridCollection[indices[j][0]],self.gridCollection[indices[j][1]],plotData,cmap='seismic',
                                    vmin = modeMin, vmax = modeMax)
                    if labels != None :
                        ax.set_xlabel(labels[indices[j][0]],fontsize=16)
                        ax.set_ylabel(labels[indices[j][1]],fontsize=16)
                plt.subplots_adjust(wspace=1/2,hspace=1/2)
                plt.savefig('../figures/mode'+str(i)+'real.png')
                plt.close()
                
                # Imaginary components of modes
                plt.figure(figsize=(10,10))
                axes = generatePlotAxes(self.dimension)
                indices = generatePlotIndices(self.dimension)
                for j in range(0,len(axes)):
                    ax = axes[j]
                    sumIndices = list(np.arange(0,self.dimension))
                    for elements in indices[j]:
                        sumIndices.remove(elements)
                    sumIndices = tuple(sumIndices)
                    plotData = np.sum(self.allInfo[self.sliceMap['eigenvectors']][i].reshape(self.grid).imag,axis=sumIndices)
                    ax.pcolormesh(self.gridCollection[indices[j][0]],self.gridCollection[indices[j][0]],plotData,cmap='seismic',
                                    vmin = modeMin, vmax = modeMax)
                    if labels != None :
                        ax.set_xlabel(labels[indices[j][0]],fontsize=16)
                        ax.set_ylabel(labels[indices[j][1]],fontsize=16)
                plt.subplots_adjust(wspace=1/2,hspace=1/2)
                plt.savefig('../figures/mode'+str(i)+'imag.png')
                plt.close()

    def basicDiagnostics(self,sortQuantity='eigenvalues'):        
        self.sliceMap = {'eigenvalues' :np.s_[:,1],
                         'amplitudes'  :np.s_[:,0],
                         'eigenvectors':np.s_[:,2:]}
        self.getAmplitudes()
        self.allInfo = np.zeros([self.rank,self.eigenvectors_DMD.shape[0]+2])*1j
        self.allInfo[self.sliceMap['amplitudes']] = self.amplitudeEnergy
        self.allInfo[self.sliceMap['eigenvalues']] = self.eigenvalues_DMD
        self.allInfo[self.sliceMap['eigenvectors']] = self.eigenvectors_DMD.T
        
        if sortQuantity == 'eigenvalues':
            self.allInfo = self.allInfo[np.argsort(self.allInfo[self.sliceMap['eigenvalues']])]
        elif sortQuantity == 'amplitudes':
            self.allInfo = self.allInfo[np.argsort(self.allInfo[self.sliceMap['amplitudes']])]
        else:
            print('warning: argument sortQuantity must be either None, eigenvalues, or amplitudes. No sorting performed.')
            return 
        
#--------------------#
# Plotting Functions #
#--------------------#
def getNumberOfAxes(dimension):
    seriesVector = np.arange(1,dimension)
    return seriesVector.sum()

def generatePlotAxes(dimension):
    rows = dimension-1
    cols = dimension-1 
    axes = []
    
    for i in range(0,rows):
        for j in range(0,cols):
            if i+j <= (dimension-2): 
                axes.append(plt.subplot2grid((rows,cols), (i,j)))
                
            else: 
                break
    return axes

def generatePlotIndices(dimension):
    indices = []
    start = 1
    for i in range(0,dimension-1):
        for j in range(start,dimension):
            indices.append([i,j])
        start += 1
    return indices
