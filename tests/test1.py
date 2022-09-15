#!/usr/bin/env python
import numpy as np
import math
from context import mdirac as md

print('----- Test 1 -----')
    
### Single valley, single spin model
a=3.15
m=0.5
d=6
epsilon=1.
Delta=2.6707803932
r0=42.003
t=1.4321384348
vf=a*t

Lunit = vf/Delta
#print(Lunit) # Angstrom

epseff=epsilon*(Delta/md.Hartree)*(Lunit/md.Bohr)
#print(epseff)

N1=66
K1=1.98*Lunit
dr=2.*math.pi/K1
#dr=a/Lunit ### For comparison with Yaroslav
cell=dr*np.array([[1.,0.],[-0.5,0.5*np.sqrt(3)]])

rmesh = md.Parallelogram(N1,cell)
kmesh = rmesh.reciprocal_cell()
kmesh.circle = kmesh.get_circle(0.3)

keld=md.interaction.Keldysh(epseff,r0/Lunit)

#Wk=md.interaction.sample_to_mesh(keld,kmesh)
Wk=md.interaction.sample_to_mesh(keld,rmesh,quadrature='fft')

inds=np.where(Wk>0.01)
print('# Wk symmetry:',Wk[inds])

Wkk=md.interaction.pairs(kmesh.circle,Wk)
E,U=md.eighT(kmesh.circle)
H=md.Hbse(E,U,Wkk)
Ex,Ux=np.linalg.eigh(H)

print('# Exciton energies:',Ex[:5])

Ex_etalone=np.array([0.79957834, 0.87849056, 0.88335509, 0.90374537, 0.91956465])
print('# Etalone: ',Ex_etalone[:5])

comparison = np.isclose(Ex[:5], Ex_etalone, atol=1e-4)
#print('Comparison: ', comparison)

if np.all(comparison):
    print('# Test 1 passed successfully')
else:
    print('# Test 1 failed')
