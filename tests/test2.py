#!/usr/bin/env python
import numpy as np
import math
from context import mdirac as md

print('----- Test 2 -----')

# Parameters from:
# Wu  et al. "Exciton band structure of monolayer MoS2"
# Phys. Rev. B 91, 075310 (2015).
a=3.193 # Angstrom
t=1.105 # eV
vf=t*a # Angstrom*eV
Delta=2*0.7925 # eV
epsilon=2.5
r0=33.875/epsilon # Angstrom

print('# Parameters from Wu  et al. Phys. Rev. B 91, 075310 (2015):')
print('# vf:', vf, 'Angstrom*eV')
print('# Delta (2*Delta_Wu):', Delta, 'eV')
print('# epsilon:', epsilon)
print('# r0:',r0, 'Angstrom')

Lunit = vf/Delta # Angstrom
r0eff=r0/Lunit
epseff=epsilon*(Delta/md.Hartree)*(Lunit/md.Bohr)

N1=400
K1=40.

print('# Numerical parameters:')
print('# N1:',N1)
print('# K1:',K1)

dr=2.*math.pi/K1 
cell=dr*np.array([[1.,0.],[-0.5,0.5*np.sqrt(3)]])

### Set mesh
print('# Setting mesh...')
rmesh = md.Parallelogram(N1,cell)
kmesh = rmesh.reciprocal_cell()
kmesh.circle = kmesh.get_circle(0.1)

print('# Calculating interaction matrix...')
keld=md.interaction.Keldysh(epseff,r0eff)
#Wk=md.interaction.sample_to_mesh(keld,kmesh)
Wk=md.interaction.sample_to_mesh(keld,rmesh,quadrature='fft')
Wkk=md.interaction.pairs(kmesh.circle,Wk)

print('# Calculating eigensystem...')
E,U=md.eighT(kmesh.circle)

print('# Setting up BSE matrix...')
H=md.Hbse(E,U,Wkk)

print('# Solving BSE eigensystem...')
Ex,Ux=np.linalg.eigh(H)

### Results & benchmarks

np.set_printoptions(precision=3)

result_Wu = np.array([0.345,0.159,0.143,0.118])
print('# Wu et al. (unconverged):',result_Wu)

result_400_40=np.array([0.33982561, 0.15352159, 0.13716585, 0.11382963])
print('# Result at N1=400, K1=40.:',result_400_40)

result_800_40=np.array([0.33982896, 0.15379718, 0.13772671, 0.10966446])
print('# Result at N1=800, K1=40.:',result_800_40)

print('# Result obtained:')
Eb=Delta*(1.-Ex[:4]) # binding energies in eV
print('Eb =',Eb)

comparison = np.isclose(Eb, result_400_40, atol=1e-3)
if np.all(comparison):
    print('# Test 2 passed successfully')
else:
    print('# Test 2 failed')    
