# copy over other imports, check cosmology 0.02235, 0.1200
# could add free params to header though currently one for mean, so trivial; add mean flux
# should figure out the xyz to yzx thing I think
# maybe save the norsd tau version
# could loadsave tng and plot here or better yet save these in data (savez anywhere?)
import os
import gc
import sys
sys.path.append("..")

import numpy as np
import asdf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM
from pmesh.pm import ParticleMesh
from nbodykit.mockmaker import gaussian_real_fields
from nbodykit.lab import cosmology

from tools import rsd_tau, density2tau, tau2deltaF, compress_asdf
from compute_power import compute_pk1d, compute_pk3d

def get_noise(k, n=0.732, k1=0.0341):
    return 1./(1 + (k/k1)**n)

def Pk_extra(k):
    return get_noise(k)

# sample directories
sim_name = "AbacusSummit_base_c000_ph006"
save_dir = "/global/cscratch1/sd/boryanah/Ly_alpha/"+sim_name

# simulation directory
Lbox = 2000. # Mpc/h
ngrid = 8000
ngrid_yz = 200
part_type = "AB"
paint_type = 'TSC' # 'CIC' # 'TSC'
z = 2.5

# load density and velocity field
f = asdf.open(save_dir+f'/z{z:.3f}/density_velocity_{paint_type}_Ndim{ngrid_yz:d}_{ngrid:d}_part{part_type}.asdf')
density = f['data']['Density'][:]
vfield = f['data']['Velocity'][:]
f.close()

# do the yzx inversion here
density = np.transpose(density, (1, 2, 0))
vfield = np.transpose(vfield, (1, 2, 0))
print(density.shape, vfield.shape)

# density is number of particles per cell and vfield is same but weighted by the 1d velocity, so divide by number of particles
vfield = vfield/density # km/s, peculiar
vfield[density == 0.] = 0.
density /= np.mean(density)

# resolution
cell_size = Lbox/ngrid
want_rsd = True

# cosmological parameters 
h = 0.6736
ombh2 = 0.02237
omch2 = 0.1200
Omega_m = (ombh2+omch2)/h**2.

# fix this with proper coordinates (though it doesn't matter too much)
cosmo = FlatLambdaCDM(H0=h*100., Om0=Omega_m, Tcmb0=2.7255)
H_z = cosmo.H(z).value
E_z = H_z/h
print("H(z) = ", H_z)

# generate noise in 1D
delta_extra = np.zeros((ngrid_yz, ngrid_yz, ngrid))
for i in range(ngrid_yz):
    for j in range(ngrid_yz):
        # create empty mesh, generate white noise (variance unity), multiply by P(k)**0.5 
        pm = ParticleMesh(Nmesh=[ngrid, 1], BoxSize=Lbox)
        delta1d_extra, _ = gaussian_real_fields(pm, Pk_extra, seed=42+j*ngrid_yz+i)
        delta1d_extra = np.asarray(delta1d_extra)
        #print(delta1d_extra.dtype, delta1d_extra.shape)
        delta_extra[i, j, :] = delta1d_extra[:, 0]

# to ensure proper normalization (should be done in nbodykit)
#delta_extra /= Lbox**1.5

# to ensure unit variance
delta_extra /= np.std(delta_extra)
sigma_extra = np.std(delta_extra)
mean_extra = np.mean(delta_extra)
print("mean, std of extra field = ", mean_extra, sigma_extra)
print("before = ", density.mean())

# lognormal field
delta_lognorm = np.exp(delta_extra - sigma_extra**2/2.) - 1.
print("mean, std of LN field = ", delta_lognorm.mean(), delta_lognorm.std())
density *= (1 + delta_lognorm)
del pm, delta_lognorm, delta_extra; gc.collect()
print("after = ", density.mean())

# generate tau
tau_0 = 1.
power = 1.6
tau = density2tau(density, tau_0, power)
del density; gc.collect()

# we have ngrid^3 cells
bins = np.linspace(0., ngrid, ngrid+1)
binc = (bins[1:] + bins[:-1]) * .5
binc *= cell_size # cMpc/h
print(bins)
print(binc)
print(tau.shape, vfield.shape, binc[0], binc[-2:], Lbox, tau.mean(), vfield.mean())

# apply RSD to tau
tau = rsd_tau(tau, vfield, binc, E_z, z, Lbox)

# cannot use fake_spectra on NERSC
"""
# convert to deltaF by rescaling
#deltaF = tau2deltaF(tau, z, mean_F=None)
deltaF = tau2deltaF_mine(tau, z, mean_F=None)

# compute power spectrum
from compute_power import compute_pk1d
k_hMpc, p1d_hMpc = compute_pk1d(deltaF, Lbox)
print("1d power = ", p1d_hMpc)
"""

# save the fgpa arrays
header = {}
header['Redshift'] = z
header['Simulation'] = sim_name
header['PaintMode'] = paint_type
header['ParticleType'] = part_type
header['RSD'] = want_rsd
#header['MeanFlux'] = 
header['DirectionsOrder'] = 'YZX'
table = {}
table['OpticalDepth'] = tau
#table['OpticalDepthNoRSD'] = tau_norsd
#table['DeltaFlux'] = deltaF

# save the FGPA maps
os.makedirs(f'{save_dir}/fgpa/z{z:.3f}', exist_ok=True)
compress_asdf(f'{save_dir}/fgpa/z{z:.3f}/maps_{paint_type}_Ndim{ngrid_yz:d}_{ngrid:d}_part{part_type}.asdf', table, header)
quit()

# compute power spectrum
k_hMpc, p1d_hMpc = compute_pk1d(deltaF, Lbox)

# plot power spectrum
plt.figure(2, figsize=(9, 7))
plt.plot(k_hMpc, p1d_hMpc*k_hMpc/np.pi, label='FGPA')


# define bins
n_k_bins = 20
n_mu_bins = 16
mu_want = [0., 0.33, 0.66, 1.]
k_hMpc, mu, p3d_hMpc, counts = compute_pk3d(deltaF, Lbox, n_k_bins, n_mu_bins)

no_nans = 0
for i in range(n_k_bins):
    if not np.any(np.isnan(mu[i])): no_nans = i
int_mu = []
for i in range(len(mu_want)):
    int_mu.append(np.argmin(np.abs(mu_want[i] - mu[no_nans])))
print(k_hMpc[:, int_mu[3]])

cosmo = cosmology.Planck15
P_L = cosmology.LinearPower(cosmo, z, transfer='EisensteinHu')

plt.figure(3, figsize=(9, 7))
plt.plot(k_hMpc[:, int_mu[0]], (p3d_hMpc)[:, int_mu[0]]/P_L.__call__(k_hMpc[:, int_mu[0]]), color='violet')
plt.plot(k_hMpc[:, int_mu[1]], (p3d_hMpc)[:, int_mu[1]]/P_L.__call__(k_hMpc[:, int_mu[1]]), color='cyan')
plt.plot(k_hMpc[:, int_mu[2]], (p3d_hMpc)[:, int_mu[2]]/P_L.__call__(k_hMpc[:, int_mu[2]]), color='yellow')
plt.plot(k_hMpc[:, int_mu[3]], (p3d_hMpc)[:, int_mu[3]]/P_L.__call__(k_hMpc[:, int_mu[3]]), color='red')
