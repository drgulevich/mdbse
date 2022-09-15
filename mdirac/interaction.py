import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
        
class Keldysh():
    def __init__(self,epseff,r0):
        self.epseff=epseff
        self.r0=r0

    def rpotential(self,absr):
        factor=np.pi/(2.*self.epseff*self.r0)
        rho=absr/self.r0
        return factor*(special.struve(0,rho)-special.yn(0,rho))
    
    def kpotential(self,absk):
        factor=2.*math.pi/self.epseff
        return factor/(absk*(1.+self.r0*absk))
        
    # Average value of kpotential in kmesh cell at k=0, ignoring self.r0
    def WK00(self,kmesh):
        #factor=8.*math.pi # circle radius dp/2
        factor=12.*math.pi*math.log(3.)/math.sqrt(3.) # integral over hexagon dp/sqrt(3.)
        return factor/(kmesh.dp*self.epseff) 
    
    # Average value of rpotential in rmesh cell at r=0
    def WR00(self,rmesh):
        WR00 = -(0.0772+math.log(0.25*rmesh.dp/self.r0))/(self.epseff*self.r0)
        return WR00
    
def pairs(subdomain,Wk):
    W=np.empty((subdomain.Np,subdomain.Np),dtype=np.float64)
    for ij0,(i0,j0) in enumerate(zip(*subdomain.inds)):
        for ij1,(i1,j1) in enumerate(zip(*subdomain.inds)):
            W[ij0,ij1]=Wk[i1-i0,j1-j0]
    return W

def bypairs(Wk,inds1,inds2):
    W=np.empty((inds1[0].size,inds2[0].size),dtype=np.float64)
    for ij0,(i0,j0) in enumerate(zip(*inds1)):
        for ij1,(i1,j1) in enumerate(zip(*inds2)):
            W[ij0,ij1]=Wk[i1-i0,j1-j0]
    return W

def sample_to_mesh(interaction,mesh,quadrature=None):
    if quadrature==None:
        kmesh=mesh
        absk=np.linalg.norm(kmesh.p,axis=2)
        WK=interaction.kpotential(absk)               
        WK[0,0] = interaction.WK00(kmesh)
        WK *= kmesh.vcell/(2.*math.pi)**2 
    elif quadrature=='fft':
        rmesh=mesh
        Rnorm = rmesh.norm(center=(0,0))            
        Wr = interaction.rpotential(Rnorm)
        Wr[0,0] = interaction.WR00(rmesh)
        WK=np.fft.fft2(Wr).real/rmesh.N1**2  # rmesh.p[0,0]=0., otherwise nonzero phase
    else:
        print('Unknown quadrature.')
        WK=np.zeros((mesh.N1,mesh.N1))
    return WK