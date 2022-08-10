import numpy as np
from pmesh.pm import ParticleMesh
from nbodykit.mockmaker import gaussian_real_fields

def get_noise(k, n=0.732, k1=0.0341):
    return 1./(1 + (k/k1)**n)

def Pk_white(k):
    return get_noise(k)

# a 8^3 mesh
pm = ParticleMesh(Nmesh=[205, 205, 205])

delta, _ = gaussian_real_fields(pm, Pk_white, seed=42)

print(np.mean(delta), np.mean(np.exp(1+1.e-4*delta)))

quit()

from nbodykit.lab import *


redshift = 0.55
cosmo = cosmology.Planck15
Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
b1 = 2.0

cat = LogNormalCatalog(Plin=Plin, nbar=3e-4, BoxSize=100., Nmesh=64, bias=b1, seed=42)

# add RSD
line_of_sight = [0,0,1]
cat['RSDPosition'] = cat['Position'] + cat['VelocityOffset'] * line_of_sight

# convert to a MeshSource, using TSC interpolation on 256^3 mesh
mesh = cat.to_mesh(window='tsc', Nmesh=256, compensated=True, position='RSDPosition')

# compute the power, specifying desired linear k-binning
r = FFTPower(mesh, mode='1d')#, dk=0.005, kmin=0.01)

# the result is stored at "power" attribute
Pk = r.power
print(Pk)

print(Pk.coords)
