import math
import cmath
import numpy as np
import itertools
import matplotlib.pyplot as plt

class Domain():
    def __init__(self):
        self.Np=None
        self.p=None    
                
    def show(self,inds=[],dpi=80,**kwargs):
        fig,ax=plt.subplots(dpi=dpi)
        ax.scatter(self.p[...,0],self.p[...,1],**kwargs)
        if inds!=[]:
            kwargs.update(c='r')
            ax.scatter(self.p[inds][...,0],self.p[inds][...,1],**kwargs)
        ax.set_aspect(1)          
        
class Subdomain(Domain):
    def __init__(self,inds):
        assert(inds[0].size==inds[1].size)
        self.Np=inds[0].size
        self.inds=inds
        
class Circle(Subdomain):
    pass
        
class Hexagon(Subdomain):
    def __init__(self,Nhexlayers):
        self.Np=None
        self.p=None
        self.Nhexlayers=Nhexlayers
        self.Nppsector=1+int(0.5*(Nhexlayers+1)*Nhexlayers) # includes center for every sector
        self.Np=6*self.Nppsector-5 # central node counted only once
        
        # Populate hexagon sectors (center is counted in every sector)
        s=[]
        ardi=np.array([[1,0], [0,1], [-1,1],[-1,0],[0,-1],[1,-1]],dtype=np.int)
        ardj=np.array([[-1,1],[-1,0],[0,-1],[1,-1],[1,0],[0,1]],dtype=np.int)                
        for (di,dj) in zip(ardi,ardj):
            ind=np.empty((self.Nppsector,2),dtype=np.int)            
            ind[0]=np.array([0,0])
            i=1
            for nlayer in range(1,Nhexlayers+1):
                for nphi in range(nlayer):
                    ind[i] = nlayer*di + nphi*dj
                    i+=1
            sector=Subdomain((ind[:,0],ind[:,1]))
            s.append(sector)
            
        # Populate hexagon (center counted only once)
        h0=s[0].inds[0]
        h1=s[0].inds[1]
        for ns in range(1,6):
            h0=np.append(h0,s[ns].inds[0][1:]) # skipping center
            h1=np.append(h1,s[ns].inds[1][1:]) # skipping center            
        self.inds=(h0,h1)
        self.s=s
    
    def fill_with(self,U0,l=0):
        Nps=U0.shape[0]
        U=np.empty((6*Nps-5,*U0.shape[1:]),dtype=U0.dtype)        
        U[0]=U0[0]
        phase=cmath.exp(2.*math.pi*1j*l/6.)
        sectorphase=1.
        normfactor=1./math.sqrt(6.)
        for ns in range(6):
            U[ns*(Nps-1)+1:ns*(Nps-1)+Nps]=sectorphase*normfactor*U0[1:]
            sectorphase*=phase            
        return U    
    
class Parallelogram(Domain):    
    def __init__(self,N1,cell):
        dp0=np.linalg.norm(cell[0])
        dp1=np.linalg.norm(cell[1])
        assert(math.isclose(dp0, dp1, rel_tol=1e-15))
        self.dp=dp0
        self.N1=N1
        self.Np=N1*N1
        self.cell = cell
        self.vcell = np.abs(np.linalg.det(cell))
        inds=N1*np.fft.fftfreq(N1)
        ii,jj=np.meshgrid(inds,inds,indexing='ij')
        self.p = np.tensordot(ii,cell[0],axes=0) + np.tensordot(jj,cell[1],axes=0)         
            
    def reciprocal(self):
        icell = 2.*np.pi*np.linalg.inv(self.cell).T
        return Parallelogram(self.N1,icell)
    
    def reciprocal_cell(self):
        icell = 2.*np.pi*np.linalg.inv(self.cell).T
        return Parallelogram(self.N1,icell/self.N1)
    
    def norm(self,center=(0,0)):
        Rc=self.p[center]
        Rnorm=np.linalg.norm(self.p-Rc,axis=2)
        for i,j in itertools.product(range(-1,2),range(-1,2)):
            r=self.p+i*self.N1*self.cell[0]+j*self.N1*self.cell[1]
            rnorm=np.linalg.norm(r-Rc,axis=2)
            inds=np.where(rnorm<Rnorm)
            Rnorm[inds]=rnorm[inds].copy()
        return Rnorm   

    def get_circle(self,fraction=0.5,center=(0,0)):
        assert(fraction<=0.5) # otherwise pair interactions will not fit the parallelogram
        diameter=fraction*0.5*math.sqrt(3.)*self.N1*self.dp
        inds = np.where(np.linalg.norm(self.p-self.p[center], axis=2) < 0.5*diameter)
        circle = Circle(inds)
        circle.p = self.p[inds]
        return circle

    def get_hexagon(self,fraction=0.5):
        assert(fraction<=0.5) # otherwise pair interactions will not fit the parallelogram
        Nhexlayers=int(fraction*0.5*(self.N1-1))
        hexagon=Hexagon(Nhexlayers)
        hexagon.p=self.p[hexagon.inds]        
        for ns in range(6):
            hexagon.s[ns].p=self.p[hexagon.s[ns].inds]
        return hexagon
