import math
import cmath
import numpy as np
from numba import jit
from numba import njit
from . import interaction # for Hbse_toeplitz

Hartree=27.211386024367243 # eV
Bohr=0.5291772105638411 # Angstrom
    
s0=np.array([[1,0],[0,1]])
sx=np.array([[0,1],[1,0]])
sy=np.array([[0,-1j],[1j,0]])
sz=np.array([[1,0], [0,-1]])    
lcp=(sx+1j*sy)/math.sqrt(2.) # Left-hand circular polarized
rcp=(sx-1j*sy)/math.sqrt(2.) # Right-hand circular polarized

# Returns: E[n], U[n,:]=U[:,n].T for massive Dirac model
# See Eq.(10) in Ref. G. P. Mikitik and Yu. V. Sharlai
# Low Temp. Phys. 34, 794 (2008); 
# https://doi.org/10.1063/1.2981389
def eighT(mesh,berry=True):
    E=np.zeros((mesh.Np,2),dtype=np.float64)
    U=np.zeros((mesh.Np,2,2),dtype=np.complex128)
    if(berry):
        func=eighT_single
    else:
        func=eighT_noberry
    for i,pi in enumerate(mesh.p):
        E[i],U[i]=func(pi)
    return E,U

@njit
def eighT_single(k):
    k2=k[0]*k[0]+k[1]*k[1]
    epsk=math.sqrt(0.25+k2)
    S = np.array([[ k[0]-1j*k[1], 0.5+epsk ],
                  [ -0.5-epsk, k[0]+1j*k[1]]])        
    factor=1./math.sqrt( epsk*(1.+2.*epsk) )        
    return np.array([-epsk,epsk]),factor*S.T

@njit
def eighT_noberry(k):
    k2=k[0]*k[0]+k[1]*k[1]
    epsk=math.sqrt(0.25+k2)
    S = np.array([[ math.sqrt(k2), 0.5+epsk ],
                  [ -0.5-epsk, math.sqrt(k2)]])        
    factor=1./math.sqrt( epsk*(1.+2.*epsk) )        
    return np.array([-epsk,epsk]),factor*S.T

@njit
def Hbse(E,U,Wkk):
    Nk=E.shape[0]
    H=np.empty((Nk,Nk),dtype=np.complex128)
    for i in range(Nk):
        vi,ci=U[i,0],U[i,1]
        for j in range(Nk):
            vj,cj=U[j,0],U[j,1]
            vv = np.vdot(vj,vi)
            cc = np.vdot(ci,cj)         
            H[i,j] = -Wkk[i,j]*cc*vv
    for i in range(Nk):
        H[i,i] += E[i,1] - E[i,0]
    return H

@njit
def Hbse_ndiag(U1,U2,Wkk):
    Nk1=U1.shape[0]
    Nk2=U2.shape[0]
    H=np.empty((Nk1,Nk2),dtype=np.complex128)
    for i in range(Nk1):
        vi,ci=U1[i,0],U1[i,1]
        for j in range(Nk2):
            vj,cj=U2[j,0],U2[j,1]
            vv = np.vdot(vj,vi)
            cc = np.vdot(ci,cj)         
            H[i,j] = -Wkk[i,j]*cc*vv
    return H

def Hbse_toeplitz(hexagon,Wk,l=0,berry=True):
    E0,U0=eighT(hexagon.s[0],berry)
    Wkk=interaction.bypairs(Wk,hexagon.s[0].inds,hexagon.s[0].inds)
    s=1./math.sqrt(6.)
    if(l!=0):
        Wkk[0,0]=1.0 # this will give artificial 0 mode for l!=0
    Wkk[0,1:]*=s
    Wkk[1:,0]*=s
    H=Hbse(E0,U0,Wkk)
    for nsector in range(1,6):
        E,U=eighT(hexagon.s[nsector],berry)
        Wkk=interaction.bypairs(Wk,hexagon.s[0].inds,hexagon.s[nsector].inds)
        Wkk[0,0]=0.
        Wkk[0,1:]*=s
        Wkk[1:,0]*=s
        Hblock=Hbse_ndiag(U0,U,Wkk)
        phase=cmath.exp(2.*math.pi*1j*l*nsector/6.)
        H+=phase*Hblock
    return H

### <c|operator|v>
def cov_matrix_elements(operator,U):
    vk,ck = U[:,0],U[:,1]
    return np.einsum('ki,ij,kj->k',ck.conj(),operator,vk)

def exciton_elements(Ux,cov):
    return np.einsum('nk,k->n',Ux,cov)

### Analytic formula for chi0 in massive Dirac model (no spins, no valleys)
# See Valera's Eq.(7) at g=1
# Also, Eq.(6) in Kotov PRB 78, 075433 (2008), multiplied by 4.
def chi0(omega):
    q=-1j*omega    
    bracket = 1./q + (1.-1./(q*q))*np.arctan(q)
    return -bracket/(8*math.pi*q)

### Evaluate chi
# factor=kmesh.vcell/(2.*math.pi)**2
# For broadening use: get_chi(omega+1j*eta)
@njit
def get_chi(factor,omega,dE,r):
    chi=np.zeros(omega.size,dtype=np.complex128)
    for i in range(dE.shape[0]):
        s=abs(r[i])**2
        chi += s/(omega-dE[i]) - s/(omega+np.conj(dE[i]))
    return factor*chi
    
@njit    
def get_dchi(factor,omega,dE,r):
    dchi=np.zeros(omega.size,dtype=np.complex128)
    for i in range(dE.shape[0]):
        s=abs(r[i])**2
        dchi += -s/(omega-dE[i])**2 + s/(omega+np.conj(dE[i]))**2
    return factor*dchi

# def get_chi_zeros_full(factor,dE,r,Xinf):
#     N=dE.shape[0]
#     Wmatrix = np.zeros((N,N),dtype=np.float64)
#     Xmatrix = np.zeros((N,N),dtype=np.float64)
#     Imatrix = np.ones((N,N),dtype=np.float64)
#     np.fill_diagonal( Wmatrix, dE)
#     np.fill_diagonal( Xmatrix, 2*dE*factor*np.abs(r)**2 )
#     M=np.block([[Wmatrix,-Xmatrix/Xinf],[Imatrix,-Wmatrix]])
#     return np.sort(np.linalg.eigvals(M))

# def get_chi_zeros(factor,dE,r,Xinf):
#     N=dE.shape[0]
#     Wmatrix = np.zeros((N,N),dtype=np.float64)
#     Xmatrix = np.zeros((N,N),dtype=np.float64)
#     Imatrix = np.ones((N,N),dtype=np.float64)
#     np.fill_diagonal( Wmatrix, dE)
#     np.fill_diagonal( Xmatrix, 2*dE*factor*np.abs(r)**2 )
#     M=Wmatrix@Wmatrix-Xmatrix@Imatrix/Xinf
#     return np.sqrt(0.j+np.sort(np.linalg.eigvals(M)))

# Zeros of the polynomial 1. + sum Fi/(w^2-wi^2)
# Example:
#     factor=kmesh.vcell/(2.*math.pi)**2
#     F = factor*2.*Ex*np.abs(rx0)**2
#     Fdiag = F/Finf
#     get_zeros(Wdiag,Fdiag)
def get_zeros(Wdiag,Fdiag):
    N=Wdiag.shape[0]
    Wmatrix = np.zeros((N,N),dtype=np.float64)
    Fmatrix = np.zeros((N,N),dtype=np.float64)
    Imatrix = np.ones((N,N),dtype=np.float64)
    np.fill_diagonal( Wmatrix, Wdiag )
    np.fill_diagonal( Fmatrix, Fdiag )
    M=Wmatrix@Wmatrix-Fmatrix@Imatrix
    return np.sqrt(0.j+np.sort(np.linalg.eigvals(M)))

# Wdiag=dE
# Xdiag = 2*dE*factor*np.abs(r)**2/Xinf
# def get_zeros(Wdiag,Xdiag):
#     N=Wdiag.shape[0]
#     Wmatrix = np.zeros((N,N),dtype=np.float64)
#     Xmatrix = np.zeros((N,N),dtype=np.float64)
#     Imatrix = np.ones((N,N),dtype=np.float64)
#     np.fill_diagonal( Wmatrix, Wdiag )
#     np.fill_diagonal( Xmatrix, Xdiag )
#     M=Wmatrix@Wmatrix-Xmatrix@Imatrix
#     return np.sqrt(0.j+np.sort(np.linalg.eigvals(M)))
