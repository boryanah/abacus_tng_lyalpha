# question: when we run out of RSD bs, what do we do w stuff that's too close? (not possible) too far (can't observe?)
# question should I match mean flux in real or redshift space
# check redshift formula
# figure out bias and beta (sherwood?)

#from dataclasses import field
import os
import gc
import sys

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from tools import rsd_tau, density2tau, tau2deltaF

from pmesh.pm import ParticleMesh
from nbodykit.mockmaker import gaussian_real_fields

def get_noise(k, n=0.732, k1=0.0341):
    return 1./(1 + (k/k1)**n)

def Pk_extra(k):
    return get_noise(k)

# sim info
ngrid = int(sys.argv[1]) #820 #410 #205
mean_density = 1.#5000. # checked independent of this
Lbox = 205. # cMpc/h
h = 0.6774
sim_name = "TNG300"
if sim_name == "TNG300":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_tng205.txt"), skiprows=1, unpack=True)
elif sim_name == "MNTG":
    snaps, _, zs, _ = np.loadtxt(os.path.expanduser("~/repos/hydrotools/hydrotools/data/snaps_illustris_mtng.txt"), skiprows=1, unpack=True)
snaps = snaps.astype(int)
snap2z_dict = {}
for i in range(len(zs)):
    snap2z_dict[snaps[i]] = zs[i]
want_rsd = True
rsd_str = "_rsd" if want_rsd else ""
fp_dm = 'fp'
paste = 'CIC'
snapshot = 29 # [2.58, 2.44, 2.32], [28, 29, 30]
redshift = snap2z_dict[snapshot]
cell_size = Lbox/ngrid
cosmo = FlatLambdaCDM(H0=h*100., Om0=0.3089, Tcmb0=2.725)
H_z = cosmo.H(redshift).value
E_z = H_z/h
print("H(z) = ", H_z)

# Ly alpha skewers directory
save_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/LyA/"

# load DM density maps
data_dir = "/n/holylfs05/LABS/hernquist_lab/Everyone/boryanah/LyA/DM_Density_field/"
if paste == "CIC":
    density = np.load(data_dir+f"density{rsd_str}_cic_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy")
    vfield = np.load(data_dir+f"vfield{rsd_str}_cic_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy")
else:
    density = np.load(data_dir+f"density{rsd_str}_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy")
    vfield = np.load(data_dir+f"vfield{rsd_str}_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy")
# density is number of particles per cell; vfield is same thing but weighted by the 1d velocity
vfield = vfield/density # km/s, peculiar
density /= np.mean(density)
density *= mean_density # TESTING!

# needs higher resolution along the line of sight (could do it for each line of sight)
#density = np.repeat
print("before the PM stuff")

# generate noise in 3D
"""
# create empty mesh, generate white noise (variance unity), multiply by P(k)**0.5 
pm = ParticleMesh(Nmesh=[ngrid, ngrid, ngrid], BoxSize=Lbox)
delta_extra, _ = gaussian_real_fields(pm, Pk_extra, seed=42)
delta_extra = np.asarray(delta_extra)
print(delta_extra.dtype, delta_extra.shape)
"""

# generate noise in 1D
delta_extra = np.zeros((ngrid, ngrid, ngrid))
for i in range(ngrid):
    for j in range(ngrid):
        # create empty mesh, generate white noise (variance unity), multiply by P(k)**0.5 
        pm = ParticleMesh(Nmesh=[ngrid, 1], BoxSize=Lbox)
        delta1d_extra, _ = gaussian_real_fields(pm, Pk_extra, seed=42+j*ngrid+i)
        delta1d_extra = np.asarray(delta1d_extra)
        #print(delta1d_extra.dtype, delta1d_extra.shape)
        delta_extra[i, j, :] = delta1d_extra[:, 0]

# to ensure proper normalization (should be done in nbodykit)
#delta_extra /= Lbox**1.5

# to ensure unit variance
delta_extra /= np.std(delta_extra)
sigma_extra = np.std(delta_extra)
mean_extra = np.mean(delta_extra)

# save tau
print("saving density")
dens_file = os.path.join(save_dir, f'noiseless_maps/density_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy')
np.save(dens_file, density)
print("before = ", density.mean())

print("mean, std of extra field = ", mean_extra, sigma_extra)
delta_lognorm = np.exp(delta_extra - sigma_extra**2/2.) - 1.
print("mean, std of LN field = ", delta_lognorm.mean(), delta_lognorm.std())
density *= (1 + delta_lognorm)
del pm, delta_lognorm, delta_extra; gc.collect()
print("after = ", density.mean())

# save tau
print("saving density noise")
dens_file = os.path.join(save_dir, f'noiseless_maps/density_noise_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy')
np.save(dens_file, density)

# generate tau
tau_0 = 1.
power = 1.6
tau = density2tau(density, tau_0, power)
del density; gc.collect()


# save tau
tau_file = os.path.join(save_dir, f'noiseless_maps/tau_real_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy')
np.save(tau_file, tau)
#tau = np.load(tau_file)

# we have ngrid^3 cells
bins = np.linspace(0., ngrid, ngrid+1) # should be 0 1 2 ... 205
binc = (bins[1:] + bins[:-1]) * .5
binc *= cell_size # cMpc/h
print(bins)
print(binc)
print(tau.shape, vfield.shape, binc[0], binc[-2:], Lbox, tau.mean(), vfield.mean())

# apply RSD to tau
tau = rsd_tau(tau, vfield, binc, E_z, redshift, Lbox)

# save tau
tau_file = os.path.join(save_dir, f'noiseless_maps/tau_redshift_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy')
np.save(tau_file, tau)

# convert to deltaF by rescaling
deltaF = tau2deltaF(tau, redshift, mean_F=None)

# compute power spectrum
from compute_power import compute_pk1d
k_hMpc, p1d_hMpc = compute_pk1d(deltaF, Lbox)
print("1d power = ", p1d_hMpc)

# save deltaF
deltaF_file = os.path.join(save_dir, f'noiseless_maps/deltaF_ngrid_{ngrid:d}_snap_{snapshot:d}_{fp_dm}.npy')
np.save(deltaF_file, deltaF)
